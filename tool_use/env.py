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

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 render=False,
                 maxSteps=1000):
        self._timeStep = 1 / 240
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._render = render
        self._maxSteps = maxSteps
        self._terminated = 0

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
        action_high = np.array([0.005, 0.005, 0.005, 0.05, 0.3])

        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high,
                                            observation_high)
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._terminated = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, 'plane.urdf'), [0, 0, -1])

        p.loadURDF(
            os.path.join(self._urdfRoot, 'table/table.urdf'), 0.5000000,
            0.00000, -.820000, 0.000000, 0.000000, 0.0, 1.0)

        xpos = 0.55 + 0.12 * random.random()
        ypos = 0 + 0.2 * random.random()
        ang = np.pi * 0.5 + np.pi * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.blockUid = p.loadURDF(
            os.path.join(self._urdfRoot, 'block.urdf'), xpos, ypos, -0.15,
            orn[0], orn[1], orn[2], orn[3])

        p.setGravity(0, 0, -10)
        self._kuka = Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self._get_observation()
        return np.array(self._observation)

    def _get_observation(self):
        self._observation = self._kuka.getObservation()
        gripper_state = p.getLinkState(self._kuka.kukaUid,
                                       self._kuka.kukaGripperIndex)
        gripper_pos = gripper_state[0]
        gripper_orn = gripper_state[1]
        block_pos, block_orn = p.getBasePositionAndOrientation(self.blockUid)

        inv_gripper_pos, inv_gripper_orn = p.invertTransform(
            gripper_pos, gripper_orn)

        block_pos_in_gripper, block_orn_in_gripper = p.multiplyTransforms(
            inv_gripper_pos, inv_gripper_orn, block_pos, block_orn)
        block_euler_in_gripper = p.getEulerFromQuaternion(block_orn_in_gripper)

        block_pos_euler_in_gripper = [
            block_pos_in_gripper[0], block_pos_in_gripper[1],
            block_pos_in_gripper[2], block_euler_in_gripper[0],
            block_euler_in_gripper[1], block_euler_in_gripper[2]
        ]

        self._observation.extend(block_pos_euler_in_gripper)
        return self._observation

    def step(self, action):
        for i in range(self._actionRepeat):
            self._kuka.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
        if self._render:
            time.sleep(self._timeStep)
        self._observation = self._get_observation()

        done = self._termination()
        npaction = np.array([action[3]])
        actionCost = np.linalg.norm(npaction) * 10.
        reward = self._reward() - actionCost

        return np.array(self._observation), reward, done, {}

    def _termination(self):
        state = p.getLinkState(self._kuka.kukaUid,
                               self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]

        if self._terminated or self._envStepCounter > self._maxSteps:
            self._observation = self._get_observation()
            return True

        maxDist = 0.005
        closestPoints = p.getClosestPoints(self._kuka.trayUid,
                                           self._kuka.kukaUid, maxDist)

        if len(closestPoints):
            self._terminated = 1

            fingerAngle = 0.3
            for i in range(100):
                graspAction = [0, 0, 0.0001, 0, fingerAngle]
                self._kuka.applyAction(graspAction)
                p.stepSimulation()
                fingerAngle = fingerAngle - (0.3 / 100)
                if fingerAngle < 0:
                    fingerAngle = 0

            for i in range(1000):
                graspAction = [0, 0, 0.001, 0, fingerAngle]
                self._kuka.applyAction(graspAction)
                p.stepSimulation()
                block_pos, block_orn = p.getBasePositionAndOrientation(
                    self.blockUid)

                if block_pos[2] > 0.23:
                    break

                state = p.getLinkState(self._kuka.kukaUid,
                                       self._kuka.kukaEndEffectorIndex)
                actualEndEffectorPos = state[0]
                if (actualEndEffectorPos[2] > 0.5):
                    break

            self._observation = self._get_observation()
            return True
        return False

    def _reward(self):
        # rewards is height of target object
        block_pos, block_orn = p.getBasePositionAndOrientation(self.blockUid)
        closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid,
                                           1000, -1,
                                           self._kuka.kukaEndEffectorIndex)

        reward = -1000

        numPt = len(closestPoints)
        if numPt > 0:
            reward = -closestPoints[0][8] * 10
        if block_pos[2] > 0.2:
            reward = reward + 10000
        return reward

    def __del__(self):
        p.disconnect()
