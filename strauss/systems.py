"""
Python code for the methods explained in Chapter 8,
'Computation of Solutions', in Partial Differential
Equations, An Introduction by Walter A. Strauss.
"""
from abc import ABC, abstractmethod
from typing import Callable, Union, Any
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Number = Union[int, float]


class System(ABC):
    """
    A class for a system in spacetime under a
    dependent variable u.

    Parameters
    ----------
    system_size: int
        The number of x values for the system. Note that the first and last
        value are used for boundary conditions.
    x_step: Number
        The size of the mesh in the x-dimension.
    t_step: Number
        The size of the mesh in the t-dimension.

    """

    def __init__(self, system_size: int, x_step: Number, t_step: Number):
        self.t_step = t_step
        self.x_step = x_step

        self.x_mesh: np.array = np.linspace(
            0, x_step * (system_size - 1), num=system_size
        )

        self.boundary_conditions = None
        self.state_history = None
        self.scheme = None

    def run(self, n_steps: int, print_state=False):
        """
        Run the simulation for n time steps.
        Parameters
        ----------
        n_steps: int
            The number of time steps to run for.
        print_state: bool, default False
            Whether to print the values of the array at each time step.
        """

        for _ in range(n_steps):
            self.step()
            if print_state:
                self.print_state()

    def step(self) -> None:
        """
        Run one step of the computation.
        """
        new_state = np.hstack((
            np.array([self.boundary_conditions[0](self.time)]),
            np.array([self.scheme(j) for j in range(1, len(self.state) - 1)]),
            np.array([self.boundary_conditions[1](self.time)]),
        ))
        self.state_history.append(new_state)

    @property
    def state(self):
        """Returns the current state of the system."""
        return self.state_history[-1]

    @property
    def time(self):
        """Returns the number of time steps that have elapsed."""
        return len(self.state_history)-1

    def print_state(self, time=None):
        """Print current system state at time t (default: current time)."""
        if time is None:
            time = self.time

        print(f"t={time} {self.state_history[time]}")

    def graph(self):
        """Graphs the current state of the system."""
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.plot(self.x_mesh, self.state, color="red")
        plt.show()

    def animate(self, num_frames: int):
        """
        Creates an animation of the system from time 0 to a specified number of frames.

        Parameters
        ----------
        num_frames: int
            The number of time frames that the animation runs for.
        """
        fig = plt.Figure()
        graph = fig.add_subplot()
        graph.set_xlim(0, max(self.x_mesh))
        history_values = [
            item for sublist in self.state_history[0:num_frames] for item in sublist
        ]  # flatten list
        graph.set_ylim(0, max(history_values))
        (line1,) = graph.plot([], [], "-", lw=2)

        def graph_frame(frame):
            line1.set_data(self.x_mesh, self.state_history[frame])

        return FuncAnimation(fig, graph_frame, frames=num_frames)

    def __add__(self, other):
        """
        Add a part to the system.
        """
        self = deepcopy(self)
        return other.__cladd__(self)
