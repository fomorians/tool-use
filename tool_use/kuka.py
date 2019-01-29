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

    def __init__(self, enable_joint_sensors=False):
        data_path = pybullet_data.getDataPath()
        kuka_path = os.path.join(data_path, 'kuka_iiwa/model.urdf')
        self.kuka_id = p.loadURDF(
            fileName=kuka_path,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True)

        if enable_joint_sensors:
            for joint_index in range(self.num_joints):
                p.enableJointForceTorqueSensor(
                    bodyUniqueId=self.kuka_id, jointIndex=joint_index)

    def reset_joint_states(self, target_values, target_velocities):
        for joint_index in range(self.num_joints):
            target_value = target_values[joint_index]
            target_velocity = target_velocities[joint_index]
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=joint_index,
                targetValue=target_value,
                targetVelocity=target_velocity)

    def get_joint_info(self):
        joint_infos = []
        for joint_index in range(self.num_joints):
            joint_info = JointInfo(*p.getJointInfo(
                bodyUniqueId=self.kuka_id, jointIndex=joint_index))
            joint_infos.append(joint_info)
        return joint_infos

    def get_joint_states(self):
        joint_states = []
        for joint_index in range(self.num_joints):
            joint_state = JointState(*p.getJointState(
                bodyUniqueId=self.kuka_id, jointIndex=joint_index))
            joint_states.append(joint_state)
        return joint_states

    def apply_joint_velocities(self, joint_velocities):
        joint_velocities = np.clip(joint_velocities, -self.max_velocity,
                                   self.max_velocity)

        for joint_index in range(self.num_joints):
            p.setJointMotorControl2(
                bodyIndex=self.kuka_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=joint_velocities[joint_index],
                force=self.max_force,
                maxVelocity=self.max_velocity)

    def apply_joint_positions(self,
                              joint_positions,
                              position_gain=0.3,
                              velocity_gain=1.0):
        joint_positions = np.clip(joint_positions, -self.joint_limits,
                                  self.joint_limits)

        for joint_index in range(self.num_joints):
            p.setJointMotorControl2(
                bodyIndex=self.kuka_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[joint_index],
                targetVelocity=0.0,
                force=self.max_force,
                maxVelocity=self.max_velocity,
                positionGain=position_gain,
                velocityGain=velocity_gain)
