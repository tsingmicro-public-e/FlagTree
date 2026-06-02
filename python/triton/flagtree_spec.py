import os
import sys


def _read_flagtree_backend():
    spec_file = os.path.join(os.path.dirname(__file__), "FLAGTREE_BACKEND")
    try:
        with open(spec_file) as f:
            return f.read().strip()
    except (FileNotFoundError, IOError):
        return ""


_FLAGTREE_BACKEND = _read_flagtree_backend()


# flagtree backend path specialization
def spec_path(path_list, exclude=()):
    if not path_list or not _FLAGTREE_BACKEND:
        return
    current_path = path_list[0].replace(os.sep, "/")
    if not current_path.endswith("/"):
        current_path += "/"
    marker = "/triton/"
    idx = current_path.find(marker)
    if idx == -1:
        return
    triton_root = current_path[:idx + len("/triton")]
    rel_path = current_path[idx + len(marker):].rstrip("/")

    # original: lookup backends/<backend>/spec/triton/<rel_path>
    backend_path = os.path.join(triton_root, "backends", _FLAGTREE_BACKEND, "spec", "triton", rel_path)
    if os.path.isdir(backend_path) and backend_path not in path_list:
        path_list.insert(0, backend_path)

    # new: lookup third_party/<backend>/python/triton/<rel_path>
    project_root = os.path.dirname(os.path.dirname(triton_root))
    third_party_path = os.path.join(project_root, "third_party", _FLAGTREE_BACKEND, "python", "triton", rel_path)
    if os.path.isdir(third_party_path) and third_party_path not in path_list:
        if exclude:
            _protect_subpackages(triton_root, exclude)
        path_list.insert(0, third_party_path)


# flagtree backend specialization
def spec(function_name: str, *args, **kwargs):
    from .runtime.driver import driver
    if hasattr(driver.active, "spec"):
        _spec = driver.active.spec
        if hasattr(_spec, function_name):
            func = getattr(_spec, function_name)
            return func(*args, **kwargs)
    return None


# flagtree backend func specialization
def spec_func(function_name: str):
    from .runtime.driver import driver
    if hasattr(driver.active, "spec"):
        _spec = driver.active.spec
        if hasattr(_spec, function_name):
            func = getattr(_spec, function_name)
            return func
    return None


class _OriginalPathFinder:
    """Forces certain subpackages to load from the original triton path,
    bypassing any overlay in __path__."""

    def __init__(self, original_triton_root, names):
        self._original = original_triton_root
        self._protected = {f"triton.{n}" for n in names}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self._protected:
            return None
        name = fullname.split(".")[-1]
        pkg_dir = os.path.join(self._original, name)
        if not os.path.isdir(pkg_dir):
            return None
        import importlib.util
        init = os.path.join(pkg_dir, "__init__.py")
        if not os.path.isfile(init):
            return None
        return importlib.util.spec_from_file_location(
            fullname, init, submodule_search_locations=[pkg_dir])


def _protect_subpackages(original_triton_root, names):
    for finder in sys.meta_path:
        if isinstance(finder, _OriginalPathFinder):
            return
    sys.meta_path.insert(0, _OriginalPathFinder(original_triton_root, names))
