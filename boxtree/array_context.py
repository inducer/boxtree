from __future__ import annotations


__copyright__ = "Copyright (C) 2022 Alexandru Fikl"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import override

from arraycontext import (
    Array,
    ArrayContext,
    ArrayOrContainerOrScalarT,
    PyOpenCLArrayContext as PyOpenCLArrayContextBase,
    ScalarLike,
    SerializedContainer,
    deserialize_container,
    rec_map_array_container,
    serialize_container,
)
from arraycontext.pytest import (
    _PytestPyOpenCLArrayContextFactoryWithClass,
    register_pytest_array_context_factory,
)
from pyopencl.algorithm import BuiltList


if TYPE_CHECKING:
    from collections.abc import Callable

    import loopy as lp
    import pytools

__doc__ = """
.. autoclass:: PyOpenCLArrayContext
.. autofunction:: dataclass_array_container
"""


# {{{ array context

def _boxtree_rec_map_container(actx: ArrayContext,
                               func: Callable[[Array], Array],
                               array: ArrayOrContainerOrScalarT,
                               allowed_types: tuple[type, ...] | None = None, *,
                               default_scalar: ScalarLike | None = None,
                               strict: bool = False) -> ArrayOrContainerOrScalarT:
    import arraycontext.impl.pyopencl.taggable_cl_array as tga

    if allowed_types is None:
        allowed_types = (tga.TaggableCLArray,)

    def _wrapper(ary: Array | None) -> Array | None:
        # NOTE: this is copied verbatim from arraycontext and this is the
        # only change to allow optional fields inside containers
        if ary is None:
            return ary

        if isinstance(ary, allowed_types):
            return func(ary)
        elif not strict and isinstance(ary, actx.array_types):
            # from warnings import warn
            # warn(f"Invoking {type(actx).__name__}.{func.__name__[1:]} with "
            #     f"{type(ary).__name__} will be unsupported in 2025. Use "
            #     "'to_tagged_cl_array' to convert instances to TaggableCLArray.",
            #     DeprecationWarning, stacklevel=2)

            return func(tga.to_tagged_cl_array(ary))
        elif np.isscalar(ary):
            if default_scalar is None:
                return ary
            else:
                return np.array(ary).dtype.type(default_scalar)
        else:
            raise TypeError(
                f"{type(actx).__name__}.{func.__name__[1:]} invoked with "
                f"an unsupported array type: got '{type(ary).__name__}', "
                f"but expected one of {allowed_types}")

    return rec_map_array_container(_wrapper, array)  # pyright: ignore[reportArgumentType]


class PyOpenCLArrayContext(PyOpenCLArrayContextBase):
    @override
    def transform_loopy_program(self, t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
        default_ep = t_unit.default_entrypoint
        options = default_ep.options

        if not (options.return_dict and options.no_numpy):
            raise ValueError("Loopy kernel passed to call_loopy must "
                    "have return_dict and no_numpy options set. "
                    "Did you use arraycontext.make_loopy_program "
                    "to create this kernel?")

        return t_unit

    # NOTE: _rec_map_container is copied from arraycontext wholesale and should
    # be kept in sync as much as possible!

    @override
    def _rec_map_container(self,
                           func: Callable[[Array], Array],
                           array: ArrayOrContainerOrScalarT,
                           allowed_types: tuple[type, ...] | None = None, *,
                           default_scalar: ScalarLike | None = None,
                           strict: bool = False) -> ArrayOrContainerOrScalarT:
        return _boxtree_rec_map_container(
            self, func, array,
            allowed_types=allowed_types,
            default_scalar=default_scalar,
            strict=strict)

# }}}


# {{{ dataclass array container

def dataclass_array_container(cls: type[pytools.T]) -> type[pytools.T]:
    """A decorator based on :func:`arraycontext.dataclass_array_container`
    that allows :class:`typing.Optional` containers.
    """

    from dataclasses import is_dataclass
    from types import UnionType
    from typing import Union, get_args, get_origin  # pyright: ignore[reportDeprecated]

    from arraycontext.container import is_array_container_type
    from arraycontext.container.dataclass import (
        _Field,
        _get_annotated_fields,
        _inject_dataclass_serialization,
    )

    assert is_dataclass(cls)

    def is_array_type(tp: type, /) -> bool:
        if tp is np.ndarray:
            from warnings import warn
            warn("Encountered 'numpy.ndarray' in a dataclass_array_container. "
                 "This is deprecated and will stop working in 2026. "
                 "If you meant an object array, use pytools.obj_array.ObjectArray. "
                 "For other uses, file an issue to discuss.",
                 DeprecationWarning, stacklevel=3)
            return True

        from arraycontext import Array
        return tp is Array or is_array_container_type(tp)

    def is_array_field(f: _Field) -> bool:
        field_type = f.type
        assert not isinstance(field_type, str)

        origin = get_origin(field_type)
        if origin in (Union, UnionType):
            return all(
                (is_array_type(arg) or arg is type(None))
                for arg in get_args(field_type))

        if not f.init:
            raise ValueError(
                    f"Field with 'init=False' not allowed: '{f.name}'")

        # NOTE:
        # * GenericAlias catches `list`, `tuple`, etc.
        # * `_BaseGenericAlias` catches `List`, `Tuple`, `Callable`, etc.
        # * `_SpecialForm` catches `Any`, `Literal`, `Optional`, etc.
        from types import GenericAlias
        from typing import (  # type: ignore[attr-defined]
            _BaseGenericAlias,
            _SpecialForm,
        )
        if isinstance(field_type, GenericAlias | _BaseGenericAlias | _SpecialForm):
            # NOTE: anything except a Union is not an array
            return False

        return is_array_type(field_type)

    from pytools import partition

    fields = _get_annotated_fields(cls)
    array_fields, non_array_fields = partition(is_array_field, fields)

    if not array_fields:
        raise ValueError(f"'{cls}' must have fields with array container type "
                "in order to use the 'dataclass_array_container' decorator")

    return _inject_dataclass_serialization(cls, array_fields, non_array_fields)

# }}}


# {{{ serialization

# NOTE: BuiltList is serialized explicitly here because pyopencl cannot depend
# on arraycontext machinery.

@serialize_container.register(BuiltList)
def serialize_built_list(obj: BuiltList) -> SerializedContainer:
    return (
        ("starts", obj.starts),
        ("lists", obj.lists),
        ("nonempty_indices", obj.nonempty_indices),
        ("compressed_indices", obj.compressed_indices),
        )


@deserialize_container.register(BuiltList)
def deserialize_built_list(template: BuiltList,
                           iterable: SerializedContainer) -> BuiltList:
    return type(template)(
        count=template.count,
        num_nonempty_lists=template.num_nonempty_lists,
        **dict(iterable))

# }}}


# {{{ pytest

def _acf() -> PyOpenCLArrayContext:
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    return PyOpenCLArrayContext(queue)


class PytestPyOpenCLArrayContextFactory(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class: ClassVar[type[ArrayContext]] = PyOpenCLArrayContext


register_pytest_array_context_factory(
    "boxtree.pyopencl",
    PytestPyOpenCLArrayContextFactory)

# }}}

# vim: fdm=marker
