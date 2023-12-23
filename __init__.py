import importlib.util
import sys
import os.path
import warnings


PREPROC_NAME = "comfyui_controlnet_aux"
where_are_we = __file__
if where_are_we.endswith(".pyc"):
    where_are_we = os.path.dirname(where_are_we)
where_are_we = os.path.dirname(os.path.dirname(where_are_we))
if not where_are_we.endswith("custom_nodes"):
    warnings.warn("Expected to be in custom_nodes, but we are in {}".format(where_are_we))
spec = importlib.util.spec_from_file_location(PREPROC_NAME, "{}/{}/__init__.py".format(where_are_we, PREPROC_NAME))
comfyui_controlnet_aux = importlib.util.module_from_spec(spec)
sys.modules[PREPROC_NAME] = comfyui_controlnet_aux 
try:
    spec.loader.exec_module(comfyui_controlnet_aux)
except Exception as e:
    raise ImportError("Failed to load comfyui_controlnet_aux: {}".format(e))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']