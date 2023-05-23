"""
Python code for the methods explained in Chapter 8,
'Computation of Solutions', in Partial Differential
Equations, An Introduction by Walter A. Strauss.
"""
from abc import ABC, abstractmethod
from typing import Callable, Union, Any

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
    phi: Callable
        A function representing the initial state
        of the system.
    system_size: int
        The number of x values for the system. Note that the first and last
        value are used for boundary conditions.
    x_step: Number
        The size of the mesh in the x-dimension.
    t_step: Number
        The size of the mesh in the t-dimension.

    """

    def __init__(self, phi: Callable, system_size: int, x_step: Number, t_step: Number):
        self.t_step = t_step
        self.x_step = x_step

        self.x_mesh: np.array = np.linspace(
            0, x_step * (system_size - 1), num=system_size
        )

        phi_ufunc = np.frompyfunc(phi, 1, 1)
        self.state: np.array = phi_ufunc(self.x_mesh)
        self.state_history = [self.state]
        self.reset()  # set system to initial state
        self.boundary_conditions = None

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
            self.state = self.step()
            if print_state:
                self.print_state()

    def step(self) -> np.array:
        """
        Run one step of the computation.

        Returns
        -------
        np.array
            The state of the system after this step.
        """
        self.time += 1
        new_state = np.array(list(map(self.template, range(len(self.state)))))
        self.state_history.append(new_state)
        return new_state

    @abstractmethod
    def template(self, j: int) -> Number:
        """
        Calculates the change in the system at a specific position j
        for one time step; i.e. for a state $u^n$ at time $n$,
        calculates $u^{n+1}_j$, the value of the dependent variable at position j
        and time n+1.

        Parameters
        ----------
        j: int
            The x-step at which the state is being calculated.

        Returns
        -------
        Number
            The value of the dependent variable at position j and time n+1.
        """
        raise NotImplementedError

    def reset(self):
        """Resets the system to its initial state."""
        self.state = self.state_history[0]
        self.state_history = [self.state_history[0]]
        self.time = 0

    # pylint: disable=invalid-name
    def add_Dirichlet_BCs(self, boundary_conditions: tuple[Callable, Callable]):
        """
        Add Dirichlet boundary conditions.

        Parameters
        ----------
        boundary_conditions: tuple(Callable, Callable)
            A pair of functions of t which the system takes at the left
            and right boundary, respectively.
            If a constant number is given, it is treated as a constant
            function.
        """

        self.boundary_conditions = tuple(
            map(constants_to_callables, boundary_conditions)
        )

    def add_Neumann_BCs(self, boundary_conditions: tuple[Callable, Callable]):
        """
        Add Neumann boundary conditions.

        Parameters
        ----------
        boundary_conditions: tuple(Callable, Callable)
            A pair of functions of t which the x-derivative of the system
            takes at the left and right boundary, respectively.
            If a constant number is given, it is treated as a constant
            function.
        """
        # we use centred differences, treating x_0 and x_j as our
        # 'ghost points'
        left_func = constants_to_callables(boundary_conditions[0])
        right_func = constants_to_callables(boundary_conditions[1])
        j = len(self.x_mesh)

        def left_BC(t):
            return -(left_func(t) * 2 * self.x_step - self.state[2])

        def right_BC(t):
            return -(right_func(t) * 2 * self.x_step - self.state[j - 2])

        self.boundary_conditions = (left_BC, right_BC)

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


class Diffusion(System):
    """
    System representing the diffusion equation
    $$u_t = u_{xx}$$.
    """

    def template(self, j: int) -> Number:
        r"""
        $$u^{n+1}_j = \frac{u^n_{j+1} - 2u^n_j + u^n_{j-1}}{(\Delta x)^2}(\Delta t) + u^n_j$$

        Using a forward difference for $u_t$ and centred second difference for $u_{xx}$.
        """
        # pylint: disable=invalid-name
        u = self.state

        # handle boundary conditions
        if j == 0:
            return self.boundary_conditions[0](self.time)
        if j + 1 == len(self.x_mesh):
            return self.boundary_conditions[1](self.time)

        return (
            (u[j + 1] - 2 * u[j] + u[j - 1]) / (self.x_step) ** 2
        ) * self.t_step + u[j]


def constants_to_callables(value: Any):
    """
    Turns a constant value into a callable.

    Parameters
    ----------
    value: Any
        If a Callable, left alone. Else, turns it into the function
        λ(j) = value.
    """
    if not isinstance(value, Callable):
        return lambda j: value
    return value
