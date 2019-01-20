import os
import gym
import time
import numpy as np
import pybullet as p

from tqdm import trange

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

        joint_position_high = [
            2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239,
            2.96705972839, 2.09439510239, 3.05432619099
        ]
        joint_velocity_high = [10] * 7
        joint_torque_high = [300] * 7
        observation_high = np.array(
            joint_position_high + joint_velocity_high + joint_torque_high,
            dtype=np.float32)
        action_high = np.array([10, 10, 10, 10, 10, 10, 10], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-observation_high, high=+observation_high)
        self.action_space = spaces.Box(low=-action_high, high=+action_high)

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

        # load randomly positioned/oriented block
        block_path = os.path.join(self.data_path, 'cube.urdf')
        block_pos_angle = np.random.sample() * np.pi * 2
        block_pos_size = np.random.uniform(0.3, 0.8)
        block_pos_height = np.random.uniform(0.2, 1.0)
        block_pos = [
            np.cos(block_pos_angle) * block_pos_size,
            np.sin(block_pos_angle) * block_pos_size,
            block_pos_height,
        ]
        block_orn = p.getQuaternionFromEuler([
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
        ])
        self.block_id = p.loadURDF(
            fileName=block_path,
            basePosition=block_pos,
            baseOrientation=block_orn,
            useFixedBase=True)

        # load kuka arm
        self.kuka = Kuka()

        target_values = [
            np.random.uniform(joint_info.jointLowerLimit,
                              joint_info.jointUpperLimit)
            for joint_info in self.kuka.get_joint_info()
        ]
        target_values = [0 for joint_index in range(self.kuka.num_joints)]
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
        observation = np.array(
            joint_positions + joint_velocities + joint_torques,
            dtype=self.observation_space.dtype)
        return observation

    def _get_reward(self):
        block_pos, block_orn = p.getBasePositionAndOrientation(self.block_id)

        end_state = p.getLinkState(self.kuka.kuka_id,
                                   self.kuka.end_effector_index)
        end_pos = end_state[4]

        reward = -np.sum(np.square(np.array(end_pos) - np.array(block_pos)))
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
