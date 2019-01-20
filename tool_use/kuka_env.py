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

    def __init__(self, should_render=False):
        self.should_render = should_render
        self.time_step = 1 / 240
        self.gravity = -9.8

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_path = os.path.abspath(
            os.path.join(dir_path, os.pardir, 'data'))
        print('KukaEnv.__file__', __file__)
        print('KukaEnv.realpath', os.path.realpath(__file__))
        print('KukaEnv.dir_path', dir_path)
        print('KukaEnv.data_path', self.data_path)
        print('KukaEnv.listdir', os.listdir(self.data_path))

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

        max_velocity = 10
        num_joints = 7
        max_force = 500  # NOTE: originally 300

        joint_position_high = np.array(
            [
                2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239,
                2.96705972839, 2.09439510239, 3.05432619099
            ],
            dtype=np.float32)
        joint_position_low = -joint_position_high

        joint_velocity_high = np.full(
            shape=7, fill_value=max_velocity, dtype=np.float32)
        joint_velocity_low = -joint_position_high

        joint_torque_high = np.full(
            shape=7, fill_value=max_force, dtype=np.float32)
        joint_torque_low = -joint_torque_high

        goal_pos_high = np.array([0.8, 0.8, 1.0], dtype=np.float32)
        goal_pos_low = np.array([0.3, 0.3, 0.2], dtype=np.float32)

        observation_high = np.concatenate([
            joint_position_high, joint_velocity_high, joint_torque_high,
            goal_pos_high
        ])
        observation_low = np.concatenate([
            joint_position_low, joint_velocity_low, joint_torque_low,
            goal_pos_low
        ])

        action_high = np.full(
            shape=num_joints, fill_value=max_velocity, dtype=np.float32)
        action_low = -action_high

        self.observation_space = spaces.Box(
            low=observation_low, high=observation_high)
        self.action_space = spaces.Box(low=action_low, high=action_high)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation()
        p.setTimeStep(self.time_step)

        # load ground plane
        p.loadURDF(
            os.path.join(self.data_path, 'plane.urdf'),
            basePosition=[0, 0, -0.05],
            useFixedBase=True)

        # load randomly positioned/oriented goal
        goal_path = os.path.join(self.data_path, 'cube.urdf')
        goal_pos_angle = np.random.sample() * np.pi * 2
        goal_pos_size = np.random.uniform(0.3, 0.8)
        goal_pos_height = np.random.uniform(0.2, 1.0)
        goal_pos = [
            np.cos(goal_pos_angle) * goal_pos_size,
            np.sin(goal_pos_angle) * goal_pos_size,
            goal_pos_height,
        ]
        goal_orn = p.getQuaternionFromEuler([
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
        ])
        self.goal_id = p.loadURDF(
            fileName=goal_path,
            basePosition=goal_pos,
            baseOrientation=goal_orn,
            useFixedBase=True)

        # load kuka arm
        self.kuka = Kuka()

        target_values = [
            np.random.uniform(joint_info.jointLowerLimit / 2,
                              joint_info.jointUpperLimit / 2)
            for joint_info in self.kuka.get_joint_info()
        ]
        self.kuka.reset_joint_states(target_values)

        p.setGravity(0, 0, self.gravity)
        p.stepSimulation()

        if self.should_render:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)

        observation = self._get_observation()
        return observation

    def _get_observation(self):
        joint_states = [(joint_state.jointPosition, joint_state.jointVelocity,
                         joint_state.appliedJointMotorTorque)
                        for joint_state in self.kuka.get_joint_state()]
        joint_positions, joint_velocities, joint_torques = zip(*joint_states)
        goal_pos, goal_orn = p.getBasePositionAndOrientation(self.goal_id)
        observation = np.array(
            joint_positions + joint_velocities + joint_torques + goal_pos,
            dtype=self.observation_space.dtype)
        return observation

    def _get_reward(self):
        goal_pos, goal_orn = p.getBasePositionAndOrientation(self.goal_id)

        end_state = p.getLinkState(self.kuka.kuka_id,
                                   self.kuka.end_effector_index)
        end_pos = end_state[4]

        reward = -np.sum(np.square(np.array(end_pos) - np.array(goal_pos)))
        return reward

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.kuka.apply_joint_velocities(action)

        p.stepSimulation()

        if self.should_render:
            time.sleep(self.time_step)

        observation = self._get_observation()
        reward = self._get_reward()
        done = False
        info = {}

        return observation, reward, done, info

    def __del__(self):
        p.disconnect()


def main():
    env = KukaEnv(should_render=True)

    episodes = 100
    max_episode_steps = 1000

    for episode in trange(episodes):
        state = env.reset()
        for step in range(max_episode_steps):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state


if __name__ == '__main__':
    main()
