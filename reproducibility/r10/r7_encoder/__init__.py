"""Round 7 sequential calibrated routed-expert encoder draft.

The package is deliberately import-safe: importing it never loads a model,
CUDA extension, checkpoint, or serving runtime. Production dependencies are
loaded only by explicit CLI subcommands.
"""

from .constants import RECIPE_MARKER, RECIPE_VERSION

__all__ = ["RECIPE_MARKER", "RECIPE_VERSION"]
