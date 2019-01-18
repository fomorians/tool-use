import os
import gym
import time
import random
import numpy as np
import pybullet as p
import pybullet_data

from gym import spaces
from gym.utils import seeding

from pybullet_envs.bullet.kuka import Kuka


class KukaEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render=False):
        self._reward_height_threshold = 0.2
        self._time_step = 1 / 240
        self._render = render

        if self._render:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        self.seed()

        observation_high = np.array(
            [2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1],
            dtype=np.float32)
        action_high = np.array(
            [0.005, 0.005, 0.005, 0.05, 0.3], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-observation_high, high=+observation_high)
        self.action_space = spaces.Box(low=-action_high, high=+action_high)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)

        urdf_root = pybullet_data.getDataPath()

        block_pos = [
            0.55 + 0.12 * random.random(), 0 + 0.2 * random.random(), -0.15
        ]
        block_pos = [
            0.4 + 0.4 * random.random(), -0.3 + 0.6 * random.random(),
            -0.05 + 1.0 * random.random()
        ]
        block_orn = p.getQuaternionFromEuler([0, 0, 0])
        block_path = os.path.join(urdf_root, 'block.urdf')
        self._block_uid = p.loadURDF(
            fileName=block_path,
            basePosition=block_pos,
            baseOrientation=block_orn,
            useFixedBase=True)

        p.setGravity(0, 0, -10)

        self._kuka = Kuka(urdfRootPath=urdf_root, timeStep=self._time_step)

        p.stepSimulation()

        observation = self._get_observation()
        return observation

    def _get_observation(self):
        observation = []

        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))

        for angle in list(euler):
            observation.append(np.cos(angle))
            observation.append(np.sin(angle))

        gripper_state = p.getLinkState(self._kuka.kukaUid,
                                       self._kuka.kukaGripperIndex)
        gripper_pos = gripper_state[0]
        gripper_orn = gripper_state[1]
        block_pos, block_orn = p.getBasePositionAndOrientation(self._block_uid)

        inv_gripper_pos, inv_gripper_orn = p.invertTransform(
            gripper_pos, gripper_orn)

        block_pos_in_gripper, block_orn_in_gripper = p.multiplyTransforms(
            inv_gripper_pos, inv_gripper_orn, block_pos, block_orn)
        block_euler_in_gripper = p.getEulerFromQuaternion(block_orn_in_gripper)

        observation.extend(list(block_pos_in_gripper))

        for angle in block_euler_in_gripper:
            observation.append(np.cos(angle))
            observation.append(np.sin(angle))

        return np.array(observation, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._kuka.applyAction(action)

        p.stepSimulation()

        if self._render:
            time.sleep(self._time_step)

        observation = self._get_observation()
        reward = self._reward()
        done = False
        info = {}

        return observation, reward, done, info

    def _reward(self):
        closest_points = p.getClosestPoints(
            bodyA=self._block_uid,
            bodyB=self._kuka.kukaUid,
            distance=1000,
            linkIndexA=-1,
            linkIndexB=self._kuka.kukaEndEffectorIndex)

        reward = 0.0

        if len(closest_points) > 0:
            closest_point = closest_points[0]
            closest_distance = closest_point[8]
            reward = -np.square(closest_distance)

        return reward

    def __del__(self):
        p.disconnect()
