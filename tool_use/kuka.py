import os
import numpy as np
import pybullet as p
import pybullet_data

from tool_use.bullet import JointInfo, JointState


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
    joint_limits = np.array(
        [
            2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239,
            2.96705972839, 2.09439510239, 3.05432619099
        ],
        dtype=np.float32)
    num_joints = 7
    max_force = 300
    max_velocity = 10
    end_effector_index = 6
    joint_indices = list(range(num_joints))

    def __init__(self):
        data_path = pybullet_data.getDataPath()
        kuka_path = os.path.join(data_path, 'kuka_iiwa/model.urdf')
        self.kuka_id = p.loadURDF(
            fileName=kuka_path,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True)

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

    def apply_joint_positions(self, joint_positions):
        joint_positions = np.clip(joint_positions, -self.max_velocity,
                                  self.max_velocity)

        p.setJointMotorControlArray(
            bodyIndex=self.kuka_id,
            jointIndices=self.joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions,
            targetVelocities=[0] * self.num_joints,
            forces=[self.max_force] * self.num_joints,
            positionGains=[0.05] * self.num_joints)
