"""Base definition for the boundary conditions of a system."""
from __future__ import annotations
from typing import Tuple, Callable, TYPE_CHECKING
from abc import abstractmethod
from functools import partial

from strauss.parts import Part

if TYPE_CHECKING:
    from strauss.systems import System


class Boundary(Part):
    """
    Defines boundary conditions.

    Parameters
    ----------
    left_boundary: Callable
    right_boundary: Callable
        A pair of functions of t which the system uses for the left
        and right boundary conditions, respectively.
        If a constant number is given, it is treated as a constant
        function.
    """

    def __init__(self, left_boundary, right_boundary):
        self.boundary_conditions: Tuple[
            Callable, Callable
        ] = self.set_boundary_conditions(
            make_constant_callable(left_boundary),
            make_constant_callable(right_boundary),
        )

    @abstractmethod
    def set_boundary_conditions(self, left_func, right_func):
        """
        Parse the boundary functions into the right boundary condition.

        Parameters
        ----------
        left_func: Callable
        right_func: Callable
            A pair of functions of t which the system uses for the left
            and right boundary conditions, respectively.

        Returns
        -------
        Tuple[Callable, Callable]
            A tuple of the left and right boundary condition. Each function
            should be a function g(sys, t) where sys is the system, and
            t is time.
        """

    def __cladd__(self, other: System) -> System:
        # we use partial function application to pass the
        # system information to the boundary condition functions
        other.boundary_conditions = (
            partial(self.boundary_conditions[0], other),
            partial(self.boundary_conditions[1], other),
        )
        return other


class Dirichlet(Boundary):
    """
    Applies Dirichlet boundary conditions; the boundary functions $g(t)$
    and $h(t)$ represent the value of the dependent variable, $u$, at
    time $t$.
    """

    def set_boundary_conditions(self, left_func, right_func):
        def dirichlet_BC(func: Callable):
            return lambda sys, t: func(t)

        return (dirichlet_BC(left_func), dirichlet_BC(right_func))


class Neumann(Boundary):
    """
    Applies Neumann boundary conditions; the boundary functions $g(t)$
    and $h(t)$ represent the time derivative of the dependent variable,
    $u_t$, at time $t$.
    """

    def set_boundary_conditions(self, left_func, right_func):
        def left_BC(sys: System, t: int):
            return -(left_func(t) * 2 * sys.x_step - sys.state[2])

        def right_BC(sys: System, t: int):
            j = len(sys.x_mesh)
            return -(right_func(t) * 2 * sys.x_step - sys.state[j - 2])

        return (left_BC, right_BC)


def make_constant_callable(value: Any):
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
