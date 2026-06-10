# Keep this module as source in frozen builds. SciPy 1.17.x contains a
# top-level cleanup block with ``del obj`` that is brittle when frozen as PYZ
# bytecode by PyInstaller.
module_collection_mode = "py"
