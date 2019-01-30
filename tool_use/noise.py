import numpy as np


class OrnsteinUhlenbeckNoise:
    def __init__(self, loc, scale, friction=0.15, dt=1e-2):
        self.loc = loc
        self.scale = scale
        self.friction = friction
        self.dt = dt
        self.x = np.zeros_like(self.loc)

    def sample(self):
        delta_mean = self.friction * (self.loc - self.x) * self.dt
        brownian_velocity = np.random.normal(
            scale=self.scale, size=self.loc.shape) * np.sqrt(self.dt)
        self.x += delta_mean + brownian_velocity
        return self.x
