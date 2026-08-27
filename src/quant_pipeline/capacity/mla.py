from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


def gib(value: float) -> int:
    return int(round(value * (1 << 30)))


@dataclass(frozen=True)
class KVLayout:
    cache_layers: int
    record_bytes_per_token_layer: int
    dcp_size: int
    block_tokens: int = 64

    def validate(self) -> None:
        for name in ("cache_layers", "record_bytes_per_token_layer", "dcp_size", "block_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def local_blocks(self, global_tokens: int) -> int:
        self.validate()
        if isinstance(global_tokens, bool) or not isinstance(global_tokens, int) or global_tokens < 0:
            raise ValueError("global_tokens must be a non-negative integer")
        global_blocks = math.ceil(global_tokens / self.block_tokens)
        return math.ceil(global_blocks / self.dcp_size)

    def bytes_per_rank(self, global_tokens: int) -> int:
        return (
            self.local_blocks(global_tokens)
            * self.block_tokens
            * self.cache_layers
            * self.record_bytes_per_token_layer
        )

    def global_tokens_from_rank_bytes(self, rank_bytes: int) -> int:
        self.validate()
        if isinstance(rank_bytes, bool) or not isinstance(rank_bytes, int) or rank_bytes < 0:
            raise ValueError("rank_bytes must be a non-negative integer")
        one_local_block = self.block_tokens * self.cache_layers * self.record_bytes_per_token_layer
        local_blocks = rank_bytes // one_local_block
        return local_blocks * self.dcp_size * self.block_tokens


@dataclass(frozen=True)
class KVComponent:
    name: str
    cache_layers: int
    record_bytes_per_token_layer: int
    shard_size: int
    block_tokens: int = 64

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("KV component name must be nonempty")
        for field in ("cache_layers", "record_bytes_per_token_layer", "shard_size", "block_tokens"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"KV component {field} must be a positive integer")

    def bytes_per_rank(self, global_tokens: int) -> int:
        self.validate()
        blocks = math.ceil(math.ceil(global_tokens / self.block_tokens) / self.shard_size)
        return blocks * self.block_tokens * self.cache_layers * self.record_bytes_per_token_layer


@dataclass(frozen=True)
class CompositeKVLayout:
    components: tuple[KVComponent, ...]
    deployment_ranks: int

    @property
    def dcp_size(self) -> int:
        return self.deployment_ranks

    def validate(self) -> None:
        if isinstance(self.deployment_ranks, bool) or not isinstance(self.deployment_ranks, int) or self.deployment_ranks < 1:
            raise ValueError("deployment_ranks must be positive")
        if not self.components:
            raise ValueError("composite KV layout requires at least one component")
        names = set()
        for component in self.components:
            component.validate()
            if component.name in names:
                raise ValueError("KV component names must be unique")
            if self.deployment_ranks % component.shard_size:
                raise ValueError("component shard_size must divide deployment_ranks")
            names.add(component.name)

    def bytes_per_rank(self, global_tokens: int) -> int:
        self.validate()
        if isinstance(global_tokens, bool) or not isinstance(global_tokens, int) or global_tokens < 0:
            raise ValueError("global_tokens must be a non-negative integer")
        return sum(component.bytes_per_rank(global_tokens) for component in self.components)

    def global_tokens_from_rank_bytes(self, rank_bytes: int) -> int:
        self.validate()
        if isinstance(rank_bytes, bool) or not isinstance(rank_bytes, int) or rank_bytes < 0:
            raise ValueError("rank_bytes must be a non-negative integer")
        low, high = 0, 1
        while self.bytes_per_rank(high) <= rank_bytes:
            high *= 2
        while low + 1 < high:
            middle = (low + high) // 2
            if self.bytes_per_rank(middle) <= rank_bytes:
                low = middle
            else:
                high = middle
        # Round down to the deployment-visible global block quantum shared by
        # every component. Mixed target/draft shard sizes require the least
        # common multiple; using the smallest quantum can overstate tokens.
        quantum = math.lcm(*(component.block_tokens * component.shard_size for component in self.components))
        return (low // quantum) * quantum


@dataclass(frozen=True)
class DeviceBudget:
    device: int
    total_hbm_bytes: int
    startup_free_bytes: int
    utilization: float
    weight_bytes: int
    non_weight_non_kv_bytes: int
    safety_margin_bytes: int

    def validate(self) -> None:
        integer_fields = (
            "device",
            "total_hbm_bytes",
            "startup_free_bytes",
            "weight_bytes",
            "non_weight_non_kv_bytes",
            "safety_margin_bytes",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.utilization, (int, float)) or not math.isfinite(float(self.utilization)):
            raise ValueError("utilization must be finite")
        if not 0 < float(self.utilization) <= 1:
            raise ValueError("utilization must be in (0, 1]")

    @property
    def usable_bytes(self) -> int:
        self.validate()
        return min(math.floor(self.total_hbm_bytes * float(self.utilization)), self.startup_free_bytes)

    @property
    def kv_available_bytes(self) -> int:
        return max(
            0,
            self.usable_bytes
            - self.weight_bytes
            - self.non_weight_non_kv_bytes
            - self.safety_margin_bytes,
        )


@dataclass(frozen=True)
class RuntimeCapacityPlan:
    layout: KVLayout | CompositeKVLayout
    devices: tuple[DeviceBudget, ...]
    target_tokens: int
    limiting_device: int
    projected_tokens: int
    target_kv_bytes_per_rank: int
    required_additional_bytes_by_device: tuple[int, ...]
    maximum_weight_bytes_for_target_by_device: tuple[int, ...]
    feasible: bool

    def to_dict(self) -> dict:
        body = asdict(self)
        body["schema"] = "quant-pipeline.glm-mla-capacity-plan.v1"
        return body


def build_capacity_plan(
    layout: KVLayout | CompositeKVLayout,
    devices: Iterable[DeviceBudget],
    *,
    target_tokens: int,
) -> RuntimeCapacityPlan:
    layout.validate()
    rows = tuple(devices)
    if not rows:
        raise ValueError("at least one device budget is required")
    if len(rows) != layout.dcp_size:
        raise ValueError("device count must equal dcp_size for this conservative planner")
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens < 1:
        raise ValueError("target_tokens must be a positive integer")
    if tuple(sorted(row.device for row in rows)) != tuple(range(len(rows))):
        raise ValueError("device IDs must be contiguous from zero")
    for row in rows:
        row.validate()

    capacities = tuple(layout.global_tokens_from_rank_bytes(row.kv_available_bytes) for row in rows)
    limiting_index = min(range(len(rows)), key=lambda index: (capacities[index], index))
    target_kv = layout.bytes_per_rank(target_tokens)
    shortfalls = tuple(max(0, target_kv - row.kv_available_bytes) for row in rows)
    weight_ceilings = tuple(
        max(
            0,
            row.usable_bytes - row.non_weight_non_kv_bytes - row.safety_margin_bytes - target_kv,
        )
        for row in rows
    )
    return RuntimeCapacityPlan(
        layout=layout,
        devices=rows,
        target_tokens=target_tokens,
        limiting_device=rows[limiting_index].device,
        projected_tokens=capacities[limiting_index],
        target_kv_bytes_per_rank=target_kv,
        required_additional_bytes_by_device=shortfalls,
        maximum_weight_bytes_for_target_by_device=weight_ceilings,
        feasible=not any(shortfalls),
    )
