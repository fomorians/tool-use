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
        self.reset()

        observation_size = len(self._get_observation())
        observation_high = np.array([100] * observation_size)
        action_high = np.array([1] * 5)

        self.observation_space = spaces.Box(
            low=-observation_high, high=+observation_high)
        self.action_space = spaces.Box(low=-action_high, high=action_high)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._time_step)

        urdf_root = pybullet_data.getDataPath()

        plane_path = os.path.join(urdf_root, 'plane.urdf')
        p.loadURDF(plane_path, [0, 0, -1])

        table_path = os.path.join(urdf_root, 'table/table.urdf')
        p.loadURDF(table_path, 0.5, 0.0, -0.82, 0.0, 0.0, 0.0, 1.0)

        xpos = 0.55 + 0.12 * random.random()
        ypos = 0 + 0.2 * random.random()
        ang = np.pi * 0.5 + np.pi * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        block_path = os.path.join(urdf_root, 'block.urdf')
        self._block_uid = p.loadURDF(block_path, xpos, ypos, -0.15, orn[0],
                                     orn[1], orn[2], orn[3])

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

        for pos in block_pos_in_gripper:
            observation.append(pos)

        for angle in block_euler_in_gripper:
            observation.append(np.cos(angle))
            observation.append(np.sin(angle))

        return np.array(observation)

    def step(self, action):
        action_scale = np.array([0.005, 0.005, 0.005, 0.05, 0.3])
        self._kuka.applyAction(action * action_scale)
        p.stepSimulation()

        if self._render:
            time.sleep(self._time_step)

        observation = self._get_observation()
        done = self._get_done()
        reward = self._reward()
        info = {}

        return observation, reward, done, info

    def _get_done(self):
        block_pos, block_orn = p.getBasePositionAndOrientation(self._block_uid)
        done = False

        if block_pos[2] > self._reward_height_threshold:
            done = True

        return done

    def _reward(self):
        block_pos, block_orn = p.getBasePositionAndOrientation(self._block_uid)
        closest_points = p.getClosestPoints(self._block_uid,
                                            self._kuka.kukaUid, 1000, -1,
                                            self._kuka.kukaEndEffectorIndex)

        reward = 0

        if len(closest_points) > 0:
            closest_dist = closest_points[0][8]
            reward = -closest_dist / 100

        if block_pos[2] > self._reward_height_threshold:
            reward = 10

        return reward

    def __del__(self):
        p.disconnect()
