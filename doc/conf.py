from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2013-21, Andreas Kloeckner"

ver_dic = {}
exec(compile(open("../boxtree/version.py").read(), "../boxtree/version.py", "exec"),
        ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
# The full version, including alpha/beta/rc tags.
release = ver_dic["VERSION_TEXT"]

intersphinx_mapping = {
        "https://docs.python.org/3/": None,
        "https://numpy.org/doc/stable/": None,
        "https://documen.tician.de/pyopencl/": None,
        "https://documen.tician.de/meshmode/": None,
        "https://documen.tician.de/pytential/": None,
        }

nitpick_ignore_regex = [
    ["py:class", r"numpy._?typing._generic_alias.ScalarType"],
]

# Some modules need to import things just so that sphinx can resolve symbols in
# type annotations. Often, we do not want these imports (e.g. of PyOpenCL) when
# in normal use (because they would introduce unintended side effects or hard
# dependencies). This flag exists so that these imports only occur during doc
# build. Since sphinx appears to resolve type hints lexically (as it should),
# this needs to be cross-module (since, e.g. an inherited arraycontext
# docstring can be read by sphinx when building meshmode, a dependent package),
# this needs a setting of the same name across all packages involved, that's
# why this name is as global-sounding as it is.
import sys
sys._BUILDING_SPHINX_DOCS = True
