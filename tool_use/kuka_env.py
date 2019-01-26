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

        max_velocity = 10
        num_joints = 7

        joint_position_high = np.array(
            [
                2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239,
                2.96705972839, 2.09439510239, 3.05432619099
            ],
            dtype=np.float32)
        # joint_position_high = np.ones(shape=num_joints * 2, dtype=np.float32)
        joint_position_low = -joint_position_high

        joint_velocities_high = np.full(
            shape=num_joints, fill_value=max_velocity, dtype=np.float32)
        joint_velocities_low = -joint_velocities_high

        action_high = np.full(
            shape=2, fill_value=max_velocity, dtype=np.float32)
        action_low = -action_high

        end_high = np.array([1, 1, 1], dtype=np.float32)
        end_low = np.array([-1, -1, 0], dtype=np.float32)

        goal_high = np.array([1, 1, 1], dtype=np.float32)
        goal_low = np.array([-1, -1, 0], dtype=np.float32)

        observation_high = np.concatenate(
            [joint_position_high, joint_velocities_high, end_high, goal_high])
        observation_low = np.concatenate(
            [joint_position_low, joint_velocities_low, end_low, goal_low])

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

        # load ground plane
        p.loadURDF(
            os.path.join(self.data_path, 'plane.urdf'),
            basePosition=[0, 0, -0.05],
            useFixedBase=True)

        # load randomly positioned/oriented goal
        goal_path = os.path.join(self.data_path, 'cube.urdf')
        goal_pos_angle = np.random.sample() * np.pi * 2
        goal_pos_height = np.random.uniform(0.05, 1.0)
        goal_pos_size = 1  # np.random.uniform(0.3, 0.8)
        goal_pos = [
            np.cos(goal_pos_angle) * goal_pos_size,
            np.sin(goal_pos_angle) * goal_pos_size,
            goal_pos_height,
        ]
        goal_orn = p.getQuaternionFromEuler([
            0,  # np.random.uniform(-np.pi, np.pi),
            0,  # np.random.uniform(-np.pi, np.pi),
            0,  # np.random.uniform(-np.pi, np.pi),
        ])
        self.goal_id = p.loadURDF(
            fileName=goal_path,
            basePosition=goal_pos,
            baseOrientation=goal_orn,
            useFixedBase=True)

        # load kuka arm
        self.kuka = Kuka()
        self.kuka.reset_joint_states([0, np.pi / 2, 0, 0, 0, 0, 0])

        p.setGravity(0, 0, self.gravity)
        p.stepSimulation()

        if self.should_render:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)

        observation = self._get_observation()
        return observation

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

    def _get_joint_positions(self):
        return np.array([
            joint_state.jointPosition
            for joint_state in self.kuka.get_joint_state()
        ])
        # joint_positions = []
        # for joint_state in self.kuka.get_joint_state():
        #     joint_positions.append(np.cos(joint_state.jointPosition))
        #     joint_positions.append(np.sin(joint_state.jointPosition))
        # return np.array(joint_positions)

    def _get_joint_velocities(self):
        return np.array([
            joint_state.jointVelocity
            for joint_state in self.kuka.get_joint_state()
        ])

    def _get_observation(self):
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        end_pos = self._get_end_pos()
        goal_pos = self._get_goal_pos()
        observation = np.concatenate(
            [joint_positions, joint_velocities, end_pos, goal_pos], axis=-1)
        return observation

    def _get_reward(self):
        delta_pos = self._get_delta_pos()
        reward = -np.sum(np.square(delta_pos))
        return reward

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        joint_velocities = np.zeros(shape=7, dtype=np.float32)
        joint_velocities[0] = action[0]
        joint_velocities[1] = action[1]
        self.kuka.apply_joint_velocities(joint_velocities)

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

    for episode in range(episodes):
        state = env.reset()
        for step in range(max_episode_steps):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state


if __name__ == '__main__':
    main()
