"""Classes for system parts."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, List

import numpy as np

if TYPE_CHECKING:
    from systems import System
    from strausstypes import Number


class Part(ABC):
    """
    Defines part of a system.

    Methods
    -------
    __cladd__
        Adds the part to the system.
    """

    @abstractmethod
    def __cladd__(self: Part, other: System) -> System:
        """
        Adds the part to the system.
        """
        raise NotImplementedError
