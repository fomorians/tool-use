import os
import gym
import time
import numpy as np
import pybullet as p

from gym import spaces
from gym.utils import seeding

from tool_use.kuka import Kuka
from tool_use.noise import OrnsteinUhlenbeckNoise


class KukaEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 should_render=False,
                 position_control=True,
                 velocity_penalty=1e-3,
                 enable_wind=False,
                 enable_blocks=False):
        self.position_control = position_control
        self.should_render = should_render
        self.time_step = 1 / 240
        self.gravity = -9.8
        self.velocity_penalty = velocity_penalty
        self.enable_wind = enable_wind
        self.enable_blocks = enable_blocks

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_path = os.path.abspath(os.path.join(dir_path, 'data'))

        if self.should_render:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=-75,
                cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0])
        else:
            p.connect(p.DIRECT)

        self.seed()

        if self.enable_wind:
            self.wind_noise = OrnsteinUhlenbeckNoise(
                loc=np.zeros(3),
                scale=np.ones(3) * 10,
                friction=0.15,
                dt=self.time_step)

        joint_position_high = Kuka.joint_limits
        joint_position_low = -joint_position_high

        joint_velocities_high = np.full(
            shape=Kuka.num_joints,
            fill_value=Kuka.max_velocity,
            dtype=np.float32)
        joint_velocities_low = -joint_velocities_high

        if self.position_control:
            action_high = Kuka.joint_limits
        else:
            action_high = np.full(
                shape=Kuka.num_joints,
                fill_value=Kuka.max_velocity,
                dtype=np.float32)
        action_low = -action_high

        joint_torques_high = np.full(
            shape=Kuka.num_joints, fill_value=Kuka.max_force, dtype=np.float32)
        joint_torques_low = -joint_torques_high

        delta_high = np.array([2, 2, 1], dtype=np.float32)
        delta_low = np.array([0, 0, 0], dtype=np.float32)

        reward_high = np.array([0.0], dtype=np.float32)
        reward_low = np.array([-4.0], dtype=np.float32)

        observation_high = np.concatenate([
            joint_position_high, joint_velocities_high, joint_torques_high,
            action_high, delta_high, reward_high
        ])
        observation_low = np.concatenate([
            joint_position_low, joint_velocities_low, joint_torques_low,
            action_low, delta_low, reward_low
        ])

        self.observation_space = spaces.Box(
            low=observation_low, high=observation_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation()
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, self.gravity)

        # load ground plane
        p.loadURDF(
            os.path.join(self.data_path, 'plane.urdf'),
            basePosition=[0, 0, -0.05],
            useFixedBase=True)

        # load kuka arm
        self.kuka = Kuka()

        # rejection sampling to find pose which does not intersect
        while True:
            joint_states = self._get_init_joint_states(randomize=False)
            joint_velocities = self._get_init_joint_velocities(randomize=False)
            self.kuka.reset_joint_states(joint_states, joint_velocities)

            p.stepSimulation()

            end_pos = self._get_end_pos()
            if end_pos[2] > 0.1:
                break

        # spawn goal block
        goal_pos = self._sample_goal_pos()
        goal_orn = self._sample_orn()
        self.goal_id = self._spawn_goal(goal_pos, goal_orn)

        self.prev_action = np.zeros(
            shape=self.action_space.shape, dtype=self.action_space.dtype)
        self.prev_reward = np.zeros(shape=1, dtype=np.float32)

        if self.should_render:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)

        observation = self._get_observation()
        return observation

    def _get_init_joint_states(self, randomize=False):
        if randomize:
            return [
                np.random.uniform(-Kuka.joint_limits[joint_index],
                                  Kuka.joint_limits[joint_index])
                for joint_index in range(Kuka.num_joints)
            ]
        else:
            return [0.0] * Kuka.num_joints

    def _get_init_joint_velocities(self, randomize=False):
        if randomize:
            return [
                np.random.uniform(-Kuka.max_velocity, Kuka.max_velocity)
                for joint_index in range(Kuka.num_joints)
            ]
        else:
            return [0.0] * Kuka.num_joints

    def _sample_hemisphere_surface(self, scale=1.0):
        u = np.random.uniform(0, 1) * scale
        v = np.random.uniform(0, 1) * scale

        rho = np.sqrt(u)
        theta = v * np.pi * 2

        return np.array([
            rho * np.cos(theta),
            rho * np.sin(theta),
            np.sqrt(max(0.0, scale - u)),
        ])

    def _get_dist(self, a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    def _sample_goal_pos(self):
        while True:
            goal_pos = np.random.uniform(low=[-1, -1, 0.1], high=[1, 1, 1.1])

            # rejection sample to avoid joints
            intersects_joints = False

            link_positions = [
                self._get_link_pos(link_index)
                for link_index in range(Kuka.num_joints)
            ]
            link_positions.append(np.zeros(3))

            for link_pos in link_positions:
                if self._get_dist(goal_pos, link_pos) < 0.2:
                    intersects_joints = True
                    break

            if not intersects_joints:
                break

        return goal_pos

    def _sample_orn(self):
        return np.array(
            p.getQuaternionFromEuler([
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-np.pi, np.pi),
            ]))

    def _get_goal_pos(self):
        goal_pos, goal_orn = p.getBasePositionAndOrientation(self.goal_id)
        goal_pos = np.array(goal_pos)
        return goal_pos

    def _get_link_pos(self, link_index):
        end_state = p.getLinkState(self.kuka.kuka_id, link_index)
        end_pos = np.array(end_state[0])
        return end_pos

    def _get_end_pos(self):
        return self._get_link_pos(self.kuka.end_effector_index)

    def _get_delta_pos(self):
        goal_pos = self._get_goal_pos()
        end_pos = self._get_end_pos()
        return goal_pos - end_pos

    def _get_joint_positions(self, joint_states):
        return np.array(
            [joint_state.jointPosition for joint_state in joint_states])

    def _get_joint_velocities(self, joint_states):
        return np.array(
            [joint_state.jointVelocity for joint_state in joint_states])

    def _get_joint_torques(self, joint_states):
        return np.array([
            joint_state.appliedJointMotorTorque for joint_state in joint_states
        ])

    def _get_observation(self):
        joint_states = self.kuka.get_joint_states()
        joint_positions = self._get_joint_positions(joint_states)
        joint_velocities = self._get_joint_velocities(joint_states)
        joint_torques = self._get_joint_torques(joint_states)
        delta_pos = self._get_delta_pos()
        observation = np.concatenate(
            [
                joint_positions, joint_velocities, joint_torques,
                self.prev_action, delta_pos, self.prev_reward
            ],
            axis=-1)
        return observation

    def _spawn_goal(self, pos, orn):
        block_path = os.path.join(self.data_path, 'goal.urdf')
        block_id = p.loadURDF(
            fileName=block_path,
            basePosition=pos,
            baseOrientation=orn,
            useFixedBase=True)
        return block_id

    def _spawn_distractor(self, pos, orn, global_scaling, velocity):
        block_path = os.path.join(self.data_path, 'distractor.urdf')
        block_id = p.loadURDF(
            fileName=block_path,
            basePosition=pos,
            baseOrientation=orn,
            globalScaling=global_scaling)
        p.resetBaseVelocity(
            objectUniqueId=block_id,
            linearVelocity=velocity,
            angularVelocity=[0, 0, 0])
        return block_id

    def _spawn_random_block(self):
        link_index = np.random.randint(Kuka.num_joints)
        link_pos = self._get_link_pos(link_index)
        block_pos = self._sample_hemisphere_surface(scale=2)
        block_orn = self._sample_orn()
        block_scale = np.random.uniform(1, 2)
        block_vel = (link_pos - block_pos) * block_scale * 5
        return self._spawn_distractor(block_pos, block_orn, block_scale,
                                      block_vel)

    def _get_reward(self):
        delta_pos = self._get_delta_pos()
        joint_states = self.kuka.get_joint_states()
        joint_velocities = self._get_joint_velocities(joint_states)
        delta_reward = -np.sum(np.square(delta_pos))
        velocity_reward = self.velocity_penalty * -np.sum(
            np.square(joint_velocities))
        reward = delta_reward + velocity_reward
        return reward

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.position_control:
            self.kuka.apply_joint_positions(action)
        else:
            self.kuka.apply_joint_velocities(action)

        # randomly apply a force to joints according to an OU process
        if self.enable_wind:
            wind_force = self.wind_noise.sample()
            self.kuka.apply_external_force(wind_force)

        # randomly spawn a block of random mass directed at link COMs
        if self.enable_blocks:
            if np.random.sample() < 0.035:
                self._spawn_random_block()

        p.stepSimulation()

        if self.should_render:
            time.sleep(self.time_step)

        observation = self._get_observation()
        reward = self._get_reward()
        done = False
        info = {}

        self.prev_action = action
        self.prev_reward = np.array([reward], dtype=np.float32)

        return observation, reward, done, info

    def __del__(self):
        p.disconnect()


def main():
    env = gym.make('KukaPositionRenderEnv-v0')
    episodes = 1000
    states = np.zeros(
        shape=(episodes, env.spec.max_episode_steps,
               env.observation_space.shape[0]))
    rewards = np.zeros(shape=(episodes, env.spec.max_episode_steps))
    for episode in range(episodes):
        env.reset()
        for step in range(env.spec.max_episode_steps):
            action = env.action_space.sample()
            observation_next, reward, done, info = env.step(action)
            states[episode, step] = observation_next
            rewards[episode, step] = reward
            if done:
                break
    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    main()
