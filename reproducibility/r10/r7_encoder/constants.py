"""Owner-locked Round 7 geometry and naming rules."""

from __future__ import annotations

from dataclasses import dataclass

RECIPE_MARKER = "CODEX_ROUND7"
RECIPE_VERSION = "tr3-v4-r7-draft-1"

FIRST_MOE_LAYER = 3
LAST_MOE_LAYER = 77
MTP_LAYER = 78
MOE_LAYERS = tuple(range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1))

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

HAD_K = 128
HAD_N = 128
TRELLIS_TILE = 16
TP_SLICE_QUANTUM = 128
MCG_MULT = 0xCBAC1FED
LDL_FACTORIZATION_POLICY = "cuda-only-no-oom-fallback-v1"
CUBLAS_WORKSPACE_POLICY = ":4096:8"
EXPERTS_IMPLEMENTATION = "eager"
HUB_KERNEL_POLICY = "disabled"

ALLOWED_BITS = (3, 4, 5)
FLOOR_BITS = 3
TARGET_BITS_NUMERATOR = 7
TARGET_BITS_DENOMINATOR = 2
TENSORS_PER_LAYER = NUM_EXPERTS * len(PROJECTIONS)
TARGET_BIT_UNITS_PER_LAYER = (
    TENSORS_PER_LAYER * TARGET_BITS_NUMERATOR // TARGET_BITS_DENOMINATOR
)
BASE_BIT_UNITS_PER_LAYER = TENSORS_PER_LAYER * FLOOR_BITS
UPGRADE_UNITS_PER_LAYER = TARGET_BIT_UNITS_PER_LAYER - BASE_BIT_UNITS_PER_LAYER

DEFAULT_SIGMA_REG = 0.025
DEFAULT_DRAWS = 12
MIN_DRAWS = 8
MAX_DRAWS = 16


@dataclass(frozen=True, order=True)
class TensorId:
    """A routed-expert tensor independent of checkpoint spelling."""

    layer: int
    expert: int
    projection: str

    def __post_init__(self) -> None:
        if self.layer not in MOE_LAYERS:
            raise ValueError(f"layer {self.layer} is not a Round 7 MoE layer")
        if not 0 <= self.expert < NUM_EXPERTS:
            raise ValueError(f"expert {self.expert} outside [0,{NUM_EXPERTS})")
        if self.projection not in PROJECTIONS:
            raise ValueError(f"unknown projection {self.projection!r}")

    @property
    def key(self) -> str:
        return f"L{self.layer:02d}/E{self.expert:03d}/{self.projection}"

    @property
    def hf_prefix(self) -> str:
        return f"model.layers.{self.layer}.mlp.experts.{self.expert}.{self.projection}"

    @property
    def k(self) -> int:
        return HIDDEN_SIZE if self.projection != "down_proj" else INTERMEDIATE_SIZE

    @property
    def n(self) -> int:
        return INTERMEDIATE_SIZE if self.projection != "down_proj" else HIDDEN_SIZE


def all_layer_tensor_ids(layer: int) -> tuple[TensorId, ...]:
    return tuple(
        TensorId(layer, expert, projection)
        for expert in range(NUM_EXPERTS)
        for projection in PROJECTIONS
    )


def validate_geometry() -> None:
    if len(MOE_LAYERS) != 75 or MOE_LAYERS[-1] != LAST_MOE_LAYER:
        raise AssertionError("MoE layer range drift")
    if TENSORS_PER_LAYER != 768:
        raise AssertionError("tensor count drift")
    if TARGET_BIT_UNITS_PER_LAYER != 2688 or UPGRADE_UNITS_PER_LAYER != 384:
        raise AssertionError("3.5-bpw rate arithmetic drift")
    for dim in (HIDDEN_SIZE, INTERMEDIATE_SIZE):
        if dim % HAD_K or dim % TRELLIS_TILE or dim % TP_SLICE_QUANTUM:
            raise AssertionError(f"dimension {dim} violates pinned alignment")


validate_geometry()
