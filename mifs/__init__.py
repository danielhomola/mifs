"""Parallelized Mutual Information based Feature Selection module.

``mifs`` is a Parallelized Mutual Information based Feature
Selection module.

"""

from .mifs import MutualInformationFeatureSelector
from .version import _check_module_dependencies, __version__

_check_module_dependencies()

# Boolean controlling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
CHECK_CACHE_VERSION = True

# list all submodules available in imblearn and version
__all__ = ['MutualInformationFeatureSelector',
           '__version__']
