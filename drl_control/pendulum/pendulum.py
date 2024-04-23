from collections import OrderedDict

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.utils import rewards

_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))


def get_model_and_assets():
    """Return the model and assets for the pendulum domain."""
    return common.read_model("pendulum.xml"), common.ASSETS


def swingup(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Return the swingup task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = SwingUp(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics=physics,
        task=task,
        time_limit=time_limit,
        **environment_kwargs,
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Pendulum domain."""

    def pole_vertical(self):
        """Return the vertical (z) component of the pole frame."""
        return self.named.data.xmat["pole", "zz"]

    def angular_velocity(self):
        """Return the angular velocity of the pole."""
        return self.named.data.qvel["hinge"]

    def pole_orientation(self):
        """Return the horizontal and vertical (x, y) components of the pole frame."""
        return self.named.data.xmat["pole", ["zz", "xz"]]


class SwingUp(base.Task):
    """Swing up the pendulum to the upright position."""

    def __init__(self, random=None):
        """Initialise the task."""
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Set the state of the environment at the start of the episode."""
        physics.named.data.qpos["hinge"] = self.random.uniform(-np.pi, np.pi)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Return the observation of the environment."""
        obs = OrderedDict()
        obs["position"] = physics.pole_orientation()
        obs["velocity"] = physics.angular_velocity()
        return obs

    def get_reward(self, physics):
        """Return the reward of the environment."""
        return rewards.tolerance(
            x=physics.pole_vertical(),
            bounds=(_COSINE_BOUND, 1),
            margin=2,
        )
