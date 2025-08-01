from importlib import metadata
from urllib.request import urlopen


_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2013-21, Andreas Kloeckner"
release = metadata.version("boxtree")
version = ".".join(release.split(".")[:2])

intersphinx_mapping = {
    "arraycontext": ("https://documen.tician.de/arraycontext", None),
    "meshmode": ("https://documen.tician.de/meshmode", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pyopencl": ("https://documen.tician.de/pyopencl", None),
    "pytential": ("https://documen.tician.de/pytential", None),
    "python": ("https://docs.python.org/3", None),
    "pytools": ("https://documen.tician.de/pytools", None),
}


sphinxconfig_missing_reference_aliases = {
    # numpy
    "NDArray": "obj:numpy.typing.NDArray",
    "np.floating": "class:numpy.floating",
    "np.dtype": "class:numpy.dtype",
    # pytools typing
    "ObjectArray1D": "obj:pytools.obj_array.ObjectArray1D",
    "obj_array.ObjectArray1D": "obj:pytools.obj_array.ObjectArray1D",
    # pyopencl typing
    "Allocator": "class:pyopencl.array.Allocator",
    "WaitList": "class:pyopencl.WaitList",
    "cl_array.Array": "class:pyopencl.array.Array",
    # arraycontext
    "Array": "class:arraycontext.typing.Array",
    "ArrayContext": "class:arraycontext.ArrayContext",
    # meshmode typing
    "Mesh": "class:meshmode.mesh.Mesh",
    # boxtree typing
    "TreeKind": "obj:boxtree.tree_build.TreeKind",
    "ExtentNorm": "obj:boxtree.tree_build.ExtentNorm",
}


def setup(app):
    app.connect("missing-reference", process_autodoc_missing_reference)  # noqa: F821
