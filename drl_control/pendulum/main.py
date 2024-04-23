import numpy as np
from dm_control import mujoco
from mujoco_viewer import MujocoViewer

from drl_control.pendulum.pendulum import swingup


class Renderer:
    def __init__(self, model, data) -> None:
        self.fig_count = 0
        self.fig_data = {}
        self.viewer = MujocoViewer(
            model=model,
            data=data,
            width=640,
            height=480,
        )
        camera = self._init_camera()
        self.viewer.cam = camera

        self._add_scalar(
            name="joint-angle", x_label="Timesteps", y_label="Joint Angle (rad)"
        )
        self._add_scalar(
            name="joint-velocity", x_label="Timesteps", y_label="Joint Velocity (rad/s)"
        )
        self._add_scalar(name="reward", x_label="Timesteps", y_label="Reward")

    def render(self):
        self.viewer.render()

    def log_scalar(self, name, value):
        self.viewer.add_data_to_line(
            line_name=name,
            line_data=value,
            fig_idx=self.fig_data[name],
        )

    @property
    def is_alive(self):
        return self.viewer.is_alive

    def _init_camera(self):
        """Initialise the camera."""
        camera = mujoco.MjvCamera()
        camera.azimuth = 87
        camera.distance = 1.93
        camera.elevation = -18.43
        camera.lookat = np.array([0.0528317, -0.00131033, 0.50788843])
        return camera

    def _add_scalar(self, name: str, x_label: str, y_label: str):
        """Add a scalar plot to the viewer."""
        self.viewer.add_line_to_fig(line_name=name, fig_idx=self.fig_count)
        fig = self.viewer.figs[self.fig_count]
        fig.title = y_label
        fig.flg_legend = True
        fig.xlabel = x_label
        fig.figurergba[0] = 0.2
        fig.figurergba[3] = 0.2
        fig.gridsize[0] = 5
        fig.gridsize[1] = 5
        self.fig_data[name] = self.fig_count
        self.fig_count += 1


if __name__ == "__main__":
    env = swingup()

    renderer = Renderer(env.physics.model._model, env.physics.data._data)

    action_spec = env.action_spec()
    timestep = env.reset()

    while not timestep.last():
        action = np.random.uniform(
            action_spec.minimum,
            action_spec.maximum,
            size=action_spec.shape,
        )
        action = 1.0
        timestep = env.step(action)

        renderer.log_scalar("joint-angle", env.physics.named.data.qpos["hinge"])
        renderer.log_scalar("joint-velocity", env.physics.named.data.qvel["hinge"])
        renderer.log_scalar("reward", timestep.reward)

        renderer.render()

        if not renderer.is_alive:
            break
