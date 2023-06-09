"""Base definition for the scheme of a system."""
from __future__ import annotations

from functools import partial
from typing import Union, List, Callable, Tuple, TYPE_CHECKING
from abc import abstractmethod

import numpy as np

from strauss.parts import Part

if TYPE_CHECKING:
    from systems import System

    Number = Union[int, float]


class Scheme(Part):
    """
    A scheme for the system; defines the initial conditions of the system
    and a rule for how it evolves over time.

    Parameters
    ----------
    initial_conditions: Callable or tuple of Callables
        The initial conditions for the system.

    Methods
    -------
    initialise
        Initialise the system according to the initial
        conditions.
    scheme
        The actual scheme of the system; returns the value
        of a specific point j in the system after 1 time step.
    """

    def __init__(
        self, initial_conditions: Union[Callable, Tuple[Callable]], **variables
    ):
        self.initial_conditions = initial_conditions
        self.variables = variables

    @abstractmethod
    def initialise(
        self, sys: System, initial_conditions: Union[Callable, Tuple[Callable]]
    ) -> List[np.array]:
        """
        Calculates enough initial time steps for the system
        that the system can then run.

        Parameters
        ----------
        sys: System
            The system in which the scheme exists.
        initial_conditions: Union[Callable, Tuple[Callable]]
            The initial conditions of the system, in derivative order
            (e.g. first position, then velocity, etc.)

        Returns
        -------
        List[np.array]
            A list of arrays of the system state for these initial time steps.
        """
        raise NotImplementedError

    @abstractmethod
    def scheme(self, sys: System, j: int) -> Number:
        """
        Calculates the change in the system at a specific position j
        for one time step; i.e. for a state $u^n$ at time $n$,
        calculates $u^{n+1}_j$, the value of the dependent variable at position j
        and time n+1.

        Parameters
        ----------
        sys: System
            The system in which the scheme exists.
        j: int
            The x-step at which the state is being calculated.

        Returns
        -------
        Number
            The value of the dependent variable at position j and time n+1.
        """
        raise NotImplementedError

    def __cladd__(self, other: System) -> System:
        """
        Adds the scheme to the system.
        """
        other.state_history = self.initialise(other, self.initial_conditions)

        # we use partial function application to pass the
        # system information to the boundary condition functions
        other.scheme = partial(self.scheme, other)

        return other


class Diffusion(Scheme):
    r"""
    System representing the diffusion equation
    $$u_t = u_{xx}$$, under the scheme

    $$u^{n+1}_j = \frac{u^n_{j+1} - 2u^n_j + u^n_{j-1}}{(\Delta x)^2}(\Delta t) + u^n_j$$

    Using a forward difference for $u_t$ and centred second difference for $u_{xx}$.
    """

    def initialise(self, sys: System, initial_conditions):
        if not callable(initial_conditions):
            raise ValueError(
                "Diffusion requires one function as its initial condition."
            )

        phi_ufunc = np.frompyfunc(initial_conditions, 1, 1)
        return [phi_ufunc(sys.x_mesh)]

    def scheme(self, sys: System, j: int) -> Number:
        u = sys.state  # pylint: disable=invalid-name
        return ((u[j + 1] - 2 * u[j] + u[j - 1]) / (sys.x_step) ** 2) * sys.t_step + u[
            j
        ]


class Wave(Scheme):
    r"""
    System representing the wave equation
    $$u_{tt} = c^2 u_{xx}$$, under the scheme

    $$u^{n+1}_j = s(u^n_{j+1} + u^n_{j-1}) + 2(1 - s)u^n_j - u^{n-1}_j$$

    Using a forward difference for $u_t$ and centred second difference for $u_{xx}$,
    and where $s = (c(\Delta t)/(\Delta x))^2$.
    """

    def initialise(self, sys: System, initial_conditions):
        if callable(initial_conditions) or len(initial_conditions) != 2:
            raise ValueError("Waves require two functions as its initial conditions.")
        wave_speed = self.variables.get("wave_speed", 0)
        self.s = ((wave_speed * sys.t_step) / sys.x_step) ** 2

        phi, psi = initial_conditions
        phi_ufunc = np.frompyfunc(phi, 1, 1)

        def psi_scheme(j):
            return (
                (self.s / 2) * (phi(j - sys.x_step) + phi(j + sys.x_step))
                + (1 - self.s) * phi(j)
                + psi(j) * sys.t_step
            )

        psi_ufunc = np.frompyfunc(psi_scheme, 1, 1)
        return [phi_ufunc(sys.x_mesh), psi_ufunc(sys.x_mesh)]

    def scheme(self, sys: System, j: int):
        # pylint: disable=invalid-name
        u = sys.state
        u_n_minus_1 = sys.state_history[-2]
        s = self.s

        return s*(u[j+1] + u[j-1]) + 2*(1-s)*u[j] - u_n_minus_1[j]