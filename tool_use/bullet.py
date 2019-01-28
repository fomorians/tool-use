import attr


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
