import os
import gym
import time
import numpy as np
import pybullet as p

from gym import spaces
from gym.utils import seeding

from tool_use.kuka import Kuka


class KukaEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 should_render=False,
                 position_control=False,
                 velocity_penalty=0.0):
        self.position_control = position_control
        self.should_render = should_render
        self.time_step = 1 / 240
        self.gravity = -9.8
        self.velocity_penalty = velocity_penalty

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

        goal_high = np.array([2, 2, 1], dtype=np.float32)
        goal_low = np.array([-2, -2, -1], dtype=np.float32)

        reward_high = np.array([0.0], dtype=np.float32)
        reward_low = np.array([-4.0], dtype=np.float32)

        observation_high = np.concatenate([
            joint_position_high, joint_velocities_high, joint_torques_high,
            goal_high, action_high, reward_high
        ])
        observation_low = np.concatenate([
            joint_position_low, joint_velocities_low, joint_torques_low,
            goal_low, action_low, reward_low
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

        # load randomly positioned/oriented goal
        goal_path = os.path.join(self.data_path, 'cube.urdf')
        goal_pos = self._get_init_goal_pos(randomize=True)
        goal_orn = self._get_init_goal_orn(randomize=True)
        self.goal_id = p.loadURDF(
            fileName=goal_path,
            basePosition=goal_pos,
            baseOrientation=goal_orn,
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

    def _get_init_goal_pos(self, randomize=False):
        if randomize:
            goal_pos_angle = np.random.uniform(-np.pi, np.pi)
            goal_pos_height = np.random.uniform(0.1, 1.0)
            goal_pos_size = np.random.uniform(0.3, 1.0)
        else:
            goal_pos_angle = -np.pi / 2
            goal_pos_height = 0.1
            goal_pos_size = 1
        return np.array([
            np.cos(goal_pos_angle) * goal_pos_size,
            np.sin(goal_pos_angle) * goal_pos_size,
            goal_pos_height,
        ])

    def _get_init_goal_orn(self, randomize=False):
        if randomize:
            return np.array(
                p.getQuaternionFromEuler([
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi, np.pi),
                    np.random.uniform(-np.pi, np.pi),
                ]))
        else:
            return np.array(p.getQuaternionFromEuler([
                0,
                0,
                0,
            ]))

    def _get_goal_pos(self):
        goal_pos, goal_orn = p.getBasePositionAndOrientation(self.goal_id)
        goal_pos = np.array(goal_pos)
        return goal_pos

    def _get_end_pos(self):
        end_state = p.getLinkState(self.kuka.kuka_id,
                                   self.kuka.end_effector_index)
        end_pos = np.array(end_state[0])
        return end_pos

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
        goal_pos = self._get_delta_pos()
        observation = np.concatenate(
            [
                joint_positions, joint_velocities, joint_torques, goal_pos,
                self.prev_action, self.prev_reward
            ],
            axis=-1)
        return observation

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
        # TODO: randomly apply a force to joints according to an OU process
        # - applyExternalForce
        # TODO: randomly spawn a block of random mass directed at joint COMs
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.position_control:
            self.kuka.apply_joint_positions(action)
        else:
            self.kuka.apply_joint_velocities(action)

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
