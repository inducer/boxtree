from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # noqa: TC002

from arraycontext import Array  # noqa: TC001

from boxtree.array_context import dataclass_array_container


@dataclass_array_container
@dataclass(frozen=True)
class MyDeviceDataRecord:
    array: Array
    obj_array: np.ndarray
    opt_array: Array | None
    value: float
