import os
import attr
import numpy as np
import pybullet as p
import pybullet_data

HALF_PI = np.pi / 2


@attr.s
class JointInfo:
    jointIndex = attr.ib()
    jointName = attr.ib()
    jointType = attr.ib()
    qIndex = attr.ib()
    uIndex = attr.ib()
    flags = attr.ib()
    jointDamping = attr.ib()
    jointFriction = attr.ib()
    jointLowerLimit = attr.ib()
    jointUpperLimit = attr.ib()
    jointMaxForce = attr.ib()
    jointMaxVelocity = attr.ib()
    linkName = attr.ib()
    jointAxis = attr.ib()
    parentFramePos = attr.ib()
    parentFrameOrn = attr.ib()
    parentIndex = attr.ib()


@attr.s
class JointState:
    jointPosition = attr.ib()
    jointVelocity = attr.ib()
    jointReactionForces = attr.ib()
    appliedJointMotorTorque = attr.ib()


class Kuka:
    """
    Joints:

    - lbr_iiwa_joint_1: Rotator Joint 1
    - lbr_iiwa_joint_2: Tilt Joint 1
    - lbr_iiwa_joint_3: Rotator Joint 2
    - lbr_iiwa_joint_4: Tilt Joint 2
    - lbr_iiwa_joint_5: Rotator Joint 3
    - lbr_iiwa_joint_6: Tilt Joint 3
    - lbr_iiwa_joint_7: End effector
    """

    def __init__(self):
        self.data_path = pybullet_data.getDataPath()

        self.kuka_path = os.path.join(self.data_path, 'kuka_iiwa/model.urdf')
        self.kuka_id = p.loadURDF(
            fileName=os.path.join(self.data_path, 'kuka_iiwa/model.urdf'),
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True)

        self.num_joints = p.getNumJoints(bodyUniqueId=self.kuka_id)
        self.joint_indices = list(range(self.num_joints))
        self.max_force = 500  # NOTE: originally 300
        self.max_velocity = 10
        self.end_effector_index = 6

    def reset_joint_states(self, target_values):
        for joint_index in range(self.num_joints):
            target_value = target_values[joint_index]
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=joint_index,
                targetValue=target_value)

    def get_joint_info(self):
        for joint_index in range(self.num_joints):
            joint_info = JointInfo(*p.getJointInfo(
                bodyUniqueId=self.kuka_id, jointIndex=joint_index))
            yield joint_info

    def get_joint_state(self):
        for joint_index in range(self.num_joints):
            joint_state = JointState(*p.getJointState(
                bodyUniqueId=self.kuka_id, jointIndex=joint_index))
            yield joint_state

    def apply_joint_velocities(self, joint_velocities):
        joint_velocities = np.clip(joint_velocities, -self.max_velocity,
                                   self.max_velocity)
        p.setJointMotorControlArray(
            bodyIndex=self.kuka_id,
            jointIndices=self.joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=joint_velocities,
            forces=[self.max_force] * self.num_joints)
