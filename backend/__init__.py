"""The triton-shared backend.

Triton's TRITON_PLUGIN_DIRS support installs this directory as
``triton.backends.triton_shared``: it symlinks it into the Triton package,
registers the matching ``triton.backends`` entry point, links the C++ plugin
into libtriton.so, and copies the triton-shared-opt executable into ``bin/``.
The modules here use relative imports, so they work under that name.
"""
