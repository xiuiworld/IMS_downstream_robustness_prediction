"""Surrogate model definitions."""

from .fusion_multitask import FusionMultiTask
from .param_mlp import ParamMLP
from .single_task import FusionSingleTask, VisualSingleTask
from .visual_baseline import VisualBaseline

__all__ = ["FusionMultiTask", "FusionSingleTask", "ParamMLP", "VisualBaseline", "VisualSingleTask"]
