from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .data import SPECIAL_TARGETS, TARGET_FIELDS


@dataclass(frozen=True)
class EncoderOutput:
    """Named representation boundary shared by legacy and component encoders."""

    persistent: torch.Tensor
    context: torch.Tensor
    combined: torch.Tensor

    def to(self, *args: Any, **kwargs: Any) -> EncoderOutput:
        """Move every component with the same semantics as :meth:`Tensor.to`."""
        return EncoderOutput(
            persistent=self.persistent.to(*args, **kwargs),
            context=self.context.to(*args, **kwargs),
            combined=self.combined.to(*args, **kwargs),
        )


class SingleVectorOutputAdapter:
    """Expose a legacy vector through the component API without changing it.

    A single-vector model has no independently learned context branch.  Its
    vector is therefore exposed as both ``persistent`` and ``combined`` while
    ``context`` is an explicit, shape-compatible zero tensor.  This rule keeps
    legacy artifacts numerically identical and makes their limitation visible.
    """

    component_names = ("persistent", "context", "combined")

    def __call__(self, embedding: torch.Tensor) -> EncoderOutput:
        if embedding.ndim != 2:
            raise ValueError("legacy embedding must have shape [batch, features]")
        return EncoderOutput(
            persistent=embedding,
            context=torch.zeros_like(embedding),
            combined=embedding,
        )


def _component_dimensions(config: dict[str, Any]) -> dict[str, int]:
    """Declare component widths before objective heads are constructed."""
    model_config = config["model"]
    output_dim = int(model_config["user_embedding_dim"])
    combined_dim = output_dim
    variant = str(model_config.get("variant", LEGACY_SINGLE_VECTOR_VARIANT))
    if variant == "slow_fast_v1":
        combined_dim = output_dim * 2
    elif (
        variant == "two_timescale_pc"
        and str(model_config.get("ablation", "fusion")) == "fusion"
        and str(model_config.get("combined_fusion", "gated_residual")) == "normalized_concat"
    ):
        combined_dim = output_dim * 2
    return {"persistent": output_dim, "context": output_dim, "combined": combined_dim}


def _target_class_count(objective: str, vocabularies: dict[str, dict[str, int]], config: dict[str, Any]) -> int | None:
    if objective == "persistent_future_category_histogram":
        return len(vocabularies["object_category"])
    if objective == "next_time_bucket":
        return len(config["targets"][objective]["edges_hours"]) - 1
    if objective == "next_elapsed_time_bucket":
        return len(config["targets"][objective]["edges_hours"]) - 1
    return None


class SingleVectorEncoder(nn.Module):
    """Structured event encoder followed by a GRU and one user-history vector."""

    def __init__(
        self,
        vocabularies: dict[str, dict[str, int]],
        continuous_dim: int,
        config: dict[str, Any],
        categorical_fields: list[str] | None = None,
    ) -> None:
        super().__init__()
        model_config = config["model"]
        # Tensor columns have a positional meaning, so their order must be
        # explicit. JSON object keys are sorted when preprocessing artifacts
        # are written and must never be used as the tensor schema.
        self.categorical_fields = list(categorical_fields or vocabularies)
        if len(self.categorical_fields) != len(set(self.categorical_fields)):
            raise ValueError("categorical_fields contains duplicate names")
        missing = set(self.categorical_fields) - set(vocabularies)
        extra = set(vocabularies) - set(self.categorical_fields)
        if missing or extra:
            raise ValueError(
                "Categorical field/vocabulary mismatch: "
                f"missing_vocabularies={sorted(missing)}, unexpected_vocabularies={sorted(extra)}"
            )
        categorical_dim = int(model_config["categorical_embedding_dim"])
        event_dim = int(model_config["event_dim"])
        hidden_dim = int(model_config["hidden_dim"])
        output_dim = int(model_config["user_embedding_dim"])
        self.component_dimensions = _component_dimensions(config)
        dropout = float(model_config["dropout"])
        self.event_dropout = float(model_config.get("event_dropout", 0.0))

        self.categorical_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(
                    num_embeddings=len(vocabularies[field]),
                    embedding_dim=categorical_dim,
                    padding_idx=0,
                )
                for field in self.categorical_fields
            }
        )
        self.categorical_projection = nn.Linear(categorical_dim, event_dim)
        self.continuous_projection = nn.Sequential(
            nn.Linear(continuous_dim, event_dim),
            nn.GELU(),
            nn.Linear(event_dim, event_dim),
        )
        self.event_normalization = nn.LayerNorm(event_dim)
        self.gru = nn.GRU(
            input_size=event_dim,
            hidden_size=hidden_dim,
            num_layers=int(model_config["gru_layers"]),
            batch_first=True,
            dropout=dropout if int(model_config["gru_layers"]) > 1 else 0.0,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )
        self.heads = nn.ModuleDict()
        self.output_adapter = SingleVectorOutputAdapter()
        self.loss_routes = {
            objective: "combined" for objective in config.get("objectives", {})
        }
        routes = model_config.get("loss_routing", {})
        for objective, field in TARGET_FIELDS.items():
            if objective in config["objectives"] and field in vocabularies:
                route = str(routes.get(objective, "combined"))
                if route not in self.component_dimensions:
                    raise ValueError(f"Invalid loss routing: {route!r}")
                self.heads[objective] = nn.Linear(
                    self.component_dimensions[route], len(vocabularies[field])
                )
        self.head_loss_kinds: dict[str, str] = {}
        for objective, kind in SPECIAL_TARGETS.items():
            if objective not in config.get("objectives", {}):
                continue
            class_count = _target_class_count(objective, vocabularies, config)
            if class_count is None or class_count <= 0:
                raise ValueError(f"Missing target class declaration for {objective}")
            route = str(routes.get(objective, "combined"))
            if route not in self.component_dimensions:
                raise ValueError(f"Invalid loss routing: {route!r}")
            self.heads[objective] = nn.Linear(self.component_dimensions[route], class_count)
            self.head_loss_kinds[objective] = kind
        self.head_loss_kinds.update({name: "classification" for name in self.heads if name not in self.head_loss_kinds})

    def _event_tensor(
        self,
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        *,
        augment: bool,
    ) -> torch.Tensor:
        category_sum = None
        for index, field in enumerate(self.categorical_fields):
            embedded = self.categorical_embeddings[field](categorical[:, :, index])
            category_sum = embedded if category_sum is None else category_sum + embedded
        event = self.categorical_projection(category_sum) + self.continuous_projection(continuous)
        event = self.event_normalization(event)
        if augment and self.training and self.event_dropout > 0:
            keep = torch.rand(event.shape[:2], device=event.device) >= self.event_dropout
            keep[:, 0] = True
            event = event * keep.unsqueeze(-1)
        return event

    @staticmethod
    def _floating_final(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Select final valid states without accelerator integer gather/scatter."""
        time_steps = sequence.shape[1]
        cpu_positions = torch.arange(time_steps, device="cpu").unsqueeze(0)
        cpu_final_positions = (lengths.detach().cpu() - 1).unsqueeze(1)
        final_mask = (cpu_positions == cpu_final_positions).to(
            device=sequence.device,
            dtype=sequence.dtype,
        )
        return (sequence * final_mask.unsqueeze(-1)).sum(dim=1)

    def encode(
        self,
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        lengths: torch.Tensor,
        augment: bool = False,
        elapsed_hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_batch(categorical, continuous, lengths)
        event = self._event_tensor(categorical, continuous, augment=augment)

        # PackedSequence has a long-standing failure mode on Apple's MPS
        # backend for larger batches. Running the GRU over the padded tensor is
        # equivalent at every valid time step because padding is appended, not
        # prepended. Select each sequence's final valid output explicitly.
        sequence_output, _ = self.gru(event)
        # Do not use Tensor.gather here on MPS. Its backward pass is a scatter
        # over the time dimension, and Apple Silicon can corrupt the saved
        # integer indices for variable-length batches. Build the selector from
        # CPU-resident length metadata and send only a floating-point mask to
        # the accelerator. Multiplication and reduction have the same forward
        # result and avoid an integer gather/scatter in both directions.
        final_hidden = self._floating_final(sequence_output, lengths)
        return self.output_projection(final_hidden)

    @staticmethod
    def _validate_batch(
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        if categorical.ndim != 3 or continuous.ndim != 3:
            raise ValueError("categorical and continuous inputs must have shape [batch, time, features]")
        if categorical.shape[:2] != continuous.shape[:2]:
            raise ValueError("categorical and continuous inputs must share batch and time dimensions")
        if lengths.ndim != 1 or lengths.shape[0] != categorical.shape[0]:
            raise ValueError("lengths must contain one value per batch element")
        cpu_lengths = lengths.detach().cpu()
        invalid = (cpu_lengths <= 0) | (cpu_lengths > categorical.shape[1])
        if bool(invalid.any()):
            bad = cpu_lengths[invalid].tolist()
            raise ValueError(
                "Invalid sequence lengths: every history must contain between 1 and "
                f"{categorical.shape[1]} events; received {bad}"
            )

    def forward(
        self,
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        lengths: torch.Tensor,
        elapsed_hours: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedding = self.encode(categorical, continuous, lengths, augment=True, elapsed_hours=elapsed_hours)
        logits = {name: head(embedding) for name, head in self.heads.items()}
        return embedding, logits

    def encode_components(
        self,
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        lengths: torch.Tensor,
        augment: bool = False,
        elapsed_hours: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """Return the legacy representation through the named output boundary."""
        return self.output_adapter(
            self.encode(categorical, continuous, lengths, augment=augment, elapsed_hours=elapsed_hours)
        )


class FactorizedPCEncoder(SingleVectorEncoder):
    """Shared-event persistent/context encoder with explicit gated fusion.

    Both recurrent branches intentionally execute on padded tensors and use the
    same floating-mask selector as the legacy encoder.  ``lengths`` remains CPU
    control metadata; no packed sequence or integer accelerator gather is used.
    """

    def __init__(self, vocabularies: dict[str, dict[str, int]], continuous_dim: int,
                 config: dict[str, Any], categorical_fields: list[str] | None = None) -> None:
        super().__init__(vocabularies, continuous_dim, config, categorical_fields)
        model_config = config["model"]
        hidden_dim = int(model_config["hidden_dim"])
        output_dim = int(model_config["user_embedding_dim"])
        layers = int(model_config["gru_layers"])
        dropout = float(model_config["dropout"]) if layers > 1 else 0.0
        # ``self.gru`` is the long-history branch inherited from the baseline.
        self.context_gru = nn.GRU(int(model_config["event_dim"]), hidden_dim,
                                  num_layers=layers, batch_first=True, dropout=dropout)
        self.context_projection = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.GELU(),
                                                 nn.LayerNorm(output_dim))
        self.persistent_update_rate = float(model_config.get("persistent_update_rate", 0.1))
        if not 0.0 < self.persistent_update_rate <= 1.0:
            raise ValueError("persistent_update_rate must be in (0, 1]")
        self.recent_history_events = int(model_config.get("recent_history_events", 16))
        if self.recent_history_events <= 0:
            raise ValueError("recent_history_events must be positive")
        self.persistent_anchor = nn.Linear(int(model_config["event_dim"]), output_dim)
        self.fusion_gate = nn.Linear(output_dim * 2, output_dim)
        self.fusion_projection = nn.Sequential(nn.Linear(output_dim * 2, output_dim),
                                               nn.GELU(), nn.LayerNorm(output_dim))
        ablation = str(model_config.get("ablation", "fusion"))
        allowed = {"fusion", "persistent_only", "context_only"}
        if ablation not in allowed:
            raise ValueError(f"Unknown factorized ablation {ablation!r}; expected {sorted(allowed)}")
        self.ablation = ablation
        routes = model_config.get("loss_routing", {})
        self.loss_routes = {
            objective: str(routes.get(objective, "combined"))
            for objective in self.heads
        }
        invalid = {name: route for name, route in self.loss_routes.items()
                   if route not in {"persistent", "context", "combined"}}
        if invalid:
            raise ValueError(f"Invalid loss routing: {invalid}")

    def encode_components(self, categorical: torch.Tensor, continuous: torch.Tensor,
                          lengths: torch.Tensor, augment: bool = False,
                          elapsed_hours: torch.Tensor | None = None) -> EncoderOutput:
        self._validate_batch(categorical, continuous, lengths)
        event = self._event_tensor(categorical, continuous, augment=augment)
        persistent_sequence, _ = self.gru(event)
        persistent_last = self._floating_final(persistent_sequence, lengths)

        positions = torch.arange(event.shape[1], device="cpu").unsqueeze(0)
        starts = (lengths.detach().cpu() - self.recent_history_events).clamp_min(0).unsqueeze(1)
        valid = positions < lengths.detach().cpu().unsqueeze(1)
        recent = (valid & (positions >= starts)).to(device=event.device, dtype=event.dtype)
        context_sequence, _ = self.context_gru(event * recent.unsqueeze(-1))
        context = self.context_projection(self._floating_final(context_sequence, lengths))

        valid_float = valid.to(device=event.device, dtype=event.dtype)
        history_mean = (event * valid_float.unsqueeze(-1)).sum(1) / lengths.to(
            device=event.device, dtype=event.dtype).unsqueeze(1)
        anchor = self.persistent_anchor(history_mean)
        persistent = self.output_projection(persistent_last)
        rate = self.persistent_update_rate
        persistent = (1.0 - rate) * anchor + rate * persistent
        pair = torch.cat((persistent, context), dim=1)
        candidate = self.fusion_projection(pair)
        gate = torch.sigmoid(self.fusion_gate(pair))
        combined = gate * candidate + (1.0 - gate) * persistent
        if self.ablation == "persistent_only":
            context, combined = torch.zeros_like(context), persistent
        elif self.ablation == "context_only":
            persistent, combined = torch.zeros_like(persistent), context
        return EncoderOutput(persistent=persistent, context=context, combined=combined)

    def encode(self, categorical: torch.Tensor, continuous: torch.Tensor,
               lengths: torch.Tensor, augment: bool = False) -> torch.Tensor:
        return self.encode_components(categorical, continuous, lengths, augment).combined

    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor,
                lengths: torch.Tensor, elapsed_hours: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.encode_components(categorical, continuous, lengths, augment=True)
        logits = {name: head(getattr(output, self.loss_routes[name]))
                  for name, head in self.heads.items()}
        return output.combined, logits


class TwoTimescalePCEncoder(SingleVectorEncoder):
    """Persistent/context encoder with explicit slow and fast state pooling.

    A shared recurrent trunk avoids interpreting duplicated recurrent capacity
    as factorization.  The persistent branch pools the complete valid history
    with a long half-life.  The context branch pools the same states with a
    short half-life and encodes only its residual relative to the slow state.
    Residual fusion retains an exact persistent path so a learned gate cannot
    silently discard all long-horizon information.

    The half-lives are architectural hypotheses over observed event order, not
    claims about real behavioral time constants.  Sequence lengths remain CPU
    control metadata and all pooling uses floating masks for Apple MPS safety.
    """

    def __init__(self, vocabularies: dict[str, dict[str, int]], continuous_dim: int,
                 config: dict[str, Any], categorical_fields: list[str] | None = None) -> None:
        super().__init__(vocabularies, continuous_dim, config, categorical_fields)
        model_config = config["model"]
        hidden_dim = int(model_config["hidden_dim"])
        output_dim = int(model_config["user_embedding_dim"])
        self.persistent_half_life_events = float(
            model_config.get("persistent_half_life_events", 32.0)
        )
        self.context_half_life_events = float(
            model_config.get("context_half_life_events", 2.0)
        )
        if not self.context_half_life_events > 0:
            raise ValueError("context_half_life_events must be positive")
        if not self.persistent_half_life_events > self.context_half_life_events:
            raise ValueError(
                "persistent_half_life_events must be greater than "
                "context_half_life_events"
            )
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )
        self.combined_fusion = str(model_config.get("combined_fusion", "gated_residual"))
        if self.combined_fusion not in {"gated_residual", "normalized_concat"}:
            raise ValueError(
                "Unknown two-timescale combined_fusion "
                f"{self.combined_fusion!r}; expected gated_residual or normalized_concat"
            )
        if self.combined_fusion == "gated_residual":
            self.fusion_gate = nn.Linear(output_dim * 2, output_dim)
            self.fusion_normalization = nn.LayerNorm(output_dim)
        ablation = str(model_config.get("ablation", "fusion"))
        allowed = {"fusion", "persistent_only", "context_only"}
        if ablation not in allowed:
            raise ValueError(
                f"Unknown two-timescale ablation {ablation!r}; expected {sorted(allowed)}"
            )
        self.ablation = ablation
        routes = model_config.get("loss_routing", {})
        self.loss_routes = {
            objective: str(routes.get(objective, "combined"))
            for objective in self.heads
        }
        invalid = {
            name: route for name, route in self.loss_routes.items()
            if route not in {"persistent", "context", "combined"}
        }
        if invalid:
            raise ValueError(f"Invalid loss routing: {invalid}")

    @staticmethod
    def _timescale_weights(
        lengths: torch.Tensor,
        time_steps: int,
        half_life_events: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if half_life_events <= 0:
            raise ValueError("half_life_events must be positive")
        positions = torch.arange(time_steps, device="cpu").unsqueeze(0)
        cpu_lengths = lengths.detach().cpu().unsqueeze(1)
        valid = positions < cpu_lengths
        age = (cpu_lengths - 1 - positions).clamp_min(0).to(torch.float32)
        weights = torch.pow(0.5, age / half_life_events) * valid.to(torch.float32)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return weights.to(device=device, dtype=dtype)

    def encode_components(self, categorical: torch.Tensor, continuous: torch.Tensor,
                          lengths: torch.Tensor, augment: bool = False,
                          elapsed_hours: torch.Tensor | None = None) -> EncoderOutput:
        self._validate_batch(categorical, continuous, lengths)
        event = self._event_tensor(categorical, continuous, augment=augment)
        sequence, _ = self.gru(event)
        slow_weights = self._timescale_weights(
            lengths,
            sequence.shape[1],
            self.persistent_half_life_events,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        fast_weights = self._timescale_weights(
            lengths,
            sequence.shape[1],
            self.context_half_life_events,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        slow_state = (sequence * slow_weights.unsqueeze(-1)).sum(dim=1)
        fast_state = (sequence * fast_weights.unsqueeze(-1)).sum(dim=1)
        persistent = self.output_projection(slow_state)
        context = self.context_projection(fast_state - slow_state)
        if self.combined_fusion == "normalized_concat":
            combined = torch.cat((F.normalize(persistent, dim=1), F.normalize(context, dim=1)), dim=1)
        else:
            gate = torch.sigmoid(self.fusion_gate(torch.cat((persistent, context), dim=1)))
            combined = self.fusion_normalization(persistent + gate * context)
        if self.ablation == "persistent_only":
            context, combined = torch.zeros_like(context), persistent
        elif self.ablation == "context_only":
            persistent, combined = torch.zeros_like(persistent), context
        return EncoderOutput(persistent=persistent, context=context, combined=combined)

    def encode(self, categorical: torch.Tensor, continuous: torch.Tensor,
               lengths: torch.Tensor, augment: bool = False) -> torch.Tensor:
        return self.encode_components(categorical, continuous, lengths, augment).combined

    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor,
                lengths: torch.Tensor, elapsed_hours: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.encode_components(categorical, continuous, lengths, augment=True)
        logits = {
            name: head(getattr(output, self.loss_routes[name]))
            for name, head in self.heads.items()
        }
        return output.combined, logits


class _IndependentEventBranch(nn.Module):
    """One private observed-event encoder used by exactly one slow/fast branch."""

    def __init__(self, vocabularies: dict[str, dict[str, int]], continuous_dim: int,
                 config: dict[str, Any], categorical_fields: list[str]) -> None:
        super().__init__()
        model_config = config["model"]
        categorical_dim = int(model_config["categorical_embedding_dim"])
        event_dim = int(model_config["event_dim"])
        hidden_dim = int(model_config["hidden_dim"])
        layers = int(model_config["gru_layers"])
        dropout = float(model_config["dropout"]) if layers > 1 else 0.0
        self.categorical_fields = list(categorical_fields)
        self.categorical_embeddings = nn.ModuleDict({
            field: nn.Embedding(len(vocabularies[field]), categorical_dim, padding_idx=0)
            for field in self.categorical_fields
        })
        self.categorical_projection = nn.Linear(categorical_dim, event_dim)
        self.continuous_projection = nn.Sequential(
            nn.Linear(continuous_dim, event_dim), nn.GELU(), nn.Linear(event_dim, event_dim)
        )
        self.event_normalization = nn.LayerNorm(event_dim)
        self.gru = nn.GRU(event_dim, hidden_dim, num_layers=layers, batch_first=True, dropout=dropout)

    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        category_sum = None
        for index, field in enumerate(self.categorical_fields):
            embedded = self.categorical_embeddings[field](categorical[:, :, index])
            category_sum = embedded if category_sum is None else category_sum + embedded
        event = self.categorical_projection(category_sum) + self.continuous_projection(continuous)
        return self.event_normalization(event)


class SlowFastV1Encoder(SingleVectorEncoder):
    """Independent elapsed-time-aware slow and causal recent-event branches."""

    def __init__(self, vocabularies: dict[str, dict[str, int]], continuous_dim: int,
                 config: dict[str, Any], categorical_fields: list[str] | None = None) -> None:
        fields = list(categorical_fields or vocabularies)
        super().__init__(vocabularies, continuous_dim, config, categorical_fields=fields)
        model_config = config["model"]
        output_dim = int(model_config["user_embedding_dim"])
        # Remove the base single-vector path. The two branches own all event
        # projections and recurrent state; no recurrent module is shared.
        del self.categorical_embeddings, self.categorical_projection
        del self.continuous_projection, self.event_normalization, self.gru, self.output_projection
        self.persistent_branch = _IndependentEventBranch(vocabularies, continuous_dim, config, fields)
        self.context_branch = _IndependentEventBranch(vocabularies, continuous_dim, config, fields)
        self.persistent_projection = nn.Sequential(
            nn.Linear(int(model_config["hidden_dim"]), output_dim), nn.GELU(), nn.LayerNorm(output_dim)
        )
        self.context_projection = nn.Sequential(
            nn.Linear(int(model_config["hidden_dim"]), output_dim), nn.GELU(), nn.LayerNorm(output_dim)
        )
        self.persistent_decay_horizon_hours = float(
            model_config.get("persistent_decay_horizon_hours", 168.0)
        )
        self.context_recent_history_events = int(model_config.get("context_recent_history_events", 16))
        if self.persistent_decay_horizon_hours <= 24.0:
            raise ValueError("persistent_decay_horizon_hours must declare a multi-day horizon")
        if self.context_recent_history_events <= 0:
            raise ValueError("context_recent_history_events must be positive")
        routes = model_config.get("loss_routing", {})
        self.loss_routes = {objective: str(routes.get(objective, "combined")) for objective in self.heads}
        invalid = {name: route for name, route in self.loss_routes.items()
                   if route not in {"persistent", "context", "combined"}}
        if invalid:
            raise ValueError(f"Invalid loss routing: {invalid}")

    def _slow_state(self, sequence: torch.Tensor, lengths: torch.Tensor,
                    elapsed_hours: torch.Tensor | None) -> torch.Tensor:
        batch, time_steps, _ = sequence.shape
        if elapsed_hours is None:
            elapsed_hours = torch.ones((batch, time_steps), device=sequence.device, dtype=sequence.dtype)
            elapsed_hours[:, 0] = 0.0
        elapsed_hours = elapsed_hours.to(device=sequence.device, dtype=sequence.dtype).clamp_min(0.0)
        decay = torch.exp(-elapsed_hours / self.persistent_decay_horizon_hours)
        valid = (torch.arange(time_steps, device="cpu").unsqueeze(0) < lengths.detach().cpu().unsqueeze(1))
        decay = torch.where(valid.to(sequence.device), decay, torch.ones_like(decay))
        decay[:, 0] = 0.0
        state = torch.zeros((batch, sequence.shape[2]), device=sequence.device, dtype=sequence.dtype)
        for position in range(time_steps):
            state = decay[:, position:position + 1] * state + (1.0 - decay[:, position:position + 1]) * sequence[:, position]
        return state

    def encode_components(self, categorical: torch.Tensor, continuous: torch.Tensor,
                          lengths: torch.Tensor, augment: bool = False,
                          elapsed_hours: torch.Tensor | None = None) -> EncoderOutput:
        self._validate_batch(categorical, continuous, lengths)
        persistent_event = self.persistent_branch(categorical, continuous)
        context_event = self.context_branch(categorical, continuous)
        if augment and self.training and self.event_dropout > 0:
            keep = (torch.rand(persistent_event.shape[:2], device=persistent_event.device) >= self.event_dropout)
            keep[:, 0] = True
            persistent_event = persistent_event * keep.unsqueeze(-1)
            context_event = context_event * keep.unsqueeze(-1)
        persistent_sequence, _ = self.persistent_branch.gru(persistent_event)
        context_mask = (torch.arange(categorical.shape[1], device="cpu").unsqueeze(0)
                        >= (lengths.detach().cpu() - self.context_recent_history_events).clamp_min(0).unsqueeze(1))
        context_mask &= torch.arange(categorical.shape[1], device="cpu").unsqueeze(0) < lengths.detach().cpu().unsqueeze(1)
        context_sequence, _ = self.context_branch.gru(
            context_event * context_mask.to(device=context_event.device, dtype=context_event.dtype).unsqueeze(-1)
        )
        slow = self.persistent_projection(self._slow_state(persistent_sequence, lengths, elapsed_hours))
        fast = self.context_projection(self._floating_final(context_sequence, lengths))
        persistent = F.normalize(slow, dim=1)
        context = F.normalize(fast, dim=1)
        combined = torch.cat((persistent, context), dim=1)
        return EncoderOutput(persistent=persistent, context=context, combined=combined)

    def encode(self, categorical: torch.Tensor, continuous: torch.Tensor,
               lengths: torch.Tensor, augment: bool = False,
               elapsed_hours: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode_components(categorical, continuous, lengths, augment, elapsed_hours).combined

    def forward(self, categorical: torch.Tensor, continuous: torch.Tensor,
                lengths: torch.Tensor, elapsed_hours: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.encode_components(categorical, continuous, lengths, augment=True, elapsed_hours=elapsed_hours)
        logits = {name: head(getattr(output, self.loss_routes[name])) for name, head in self.heads.items()}
        return output.combined, logits


class CausalTransformerPCEncoder(TwoTimescalePCEncoder):
    """Two-timescale components over a small causal self-attention trunk.

    This variant isolates the sequence-model hypothesis: it retains the same
    slow/fast pooling, residual context definition, fusion, heads, and loss
    routing as :class:`TwoTimescalePCEncoder`, but replaces its GRU with causal
    self-attention. Observed continuous time-gap features remain part of every
    event, while a fixed relative event-order bias discourages indiscriminate
    attention to distant history without preventing it.

    Padding metadata stays on CPU until converted to floating attention and
    pooling masks. No packed sequence or integer final-state gather is used.
    """

    def __init__(self, vocabularies: dict[str, dict[str, int]], continuous_dim: int,
                 config: dict[str, Any], categorical_fields: list[str] | None = None) -> None:
        super().__init__(vocabularies, continuous_dim, config, categorical_fields)
        model_config = config["model"]
        event_dim = int(model_config["event_dim"])
        output_dim = int(model_config["user_embedding_dim"])
        heads = int(model_config.get("transformer_heads", 4))
        layers = int(model_config.get("transformer_layers", 2))
        feedforward_dim = int(model_config.get("transformer_feedforward_dim", 2 * event_dim))
        self.attention_half_life_events = float(
            model_config.get("attention_half_life_events", 16.0)
        )
        self.max_sequence_length = int(config.get("data", {}).get("max_sequence_length", 64))
        if event_dim % heads:
            raise ValueError("event_dim must be divisible by transformer_heads")
        if layers <= 0 or feedforward_dim <= 0 or self.max_sequence_length <= 0:
            raise ValueError("transformer layers, feedforward dimension, and sequence length must be positive")
        if self.attention_half_life_events <= 0:
            raise ValueError("attention_half_life_events must be positive")

        # Remove the recurrent trunk inherited solely to reuse the common event,
        # component, routing, and fusion contracts. It is not part of this
        # model's parameter count or forward graph.
        del self.gru
        self.position_embedding = nn.Embedding(self.max_sequence_length, event_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=event_dim,
                nhead=heads,
                dim_feedforward=feedforward_dim,
                dropout=float(model_config["dropout"]),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        ])
        self.transformer_normalization = nn.LayerNorm(event_dim)
        # The recurrent parent projection consumes hidden_dim; the transformer
        # emits event_dim. Rebuild only this projection and keep output semantics.
        self.output_projection = nn.Sequential(
            nn.Linear(event_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(event_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def _transformer_sequence(
        self, event: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        time_steps = event.shape[1]
        if time_steps > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {time_steps} exceeds configured maximum {self.max_sequence_length}"
            )
        cpu_positions = torch.arange(time_steps, device="cpu")
        position_ids = cpu_positions.to(device=event.device)
        encoded = event + self.position_embedding(position_ids).unsqueeze(0)

        # Query row i may attend only to key columns j <= i. The additive
        # relative-age penalty is finite for all permitted keys, so distant
        # events remain retrievable when their content is useful.
        age = cpu_positions.unsqueeze(1) - cpu_positions.unsqueeze(0)
        future = age < 0
        attention_mask = -math.log(2.0) * age.clamp_min(0).to(torch.float32)
        attention_mask = attention_mask / self.attention_half_life_events
        attention_mask[future] = float("-inf")
        attention_mask = attention_mask.to(device=event.device, dtype=event.dtype)

        valid = cpu_positions.unsqueeze(0) < lengths.detach().cpu().unsqueeze(1)
        padding_mask = torch.zeros(valid.shape, dtype=torch.float32, device="cpu")
        padding_mask[~valid] = float("-inf")
        padding_mask = padding_mask.to(device=event.device, dtype=event.dtype)
        valid_device = valid.to(device=event.device).unsqueeze(-1)
        sequence = encoded
        for layer in self.transformer_layers:
            sequence = layer(
                sequence,
                src_mask=attention_mask,
                src_key_padding_mask=padding_mask,
            )
            # Some attention kernels emit NaNs for padded query rows. Clear
            # them after every layer; otherwise a later layer can consume NaN
            # keys/values before its padding mask is applied.
            sequence = torch.where(valid_device, sequence, torch.zeros_like(sequence))
        sequence = self.transformer_normalization(sequence)
        return torch.where(valid_device, sequence, torch.zeros_like(sequence))

    def encode_components(self, categorical: torch.Tensor, continuous: torch.Tensor,
                          lengths: torch.Tensor, augment: bool = False,
                          elapsed_hours: torch.Tensor | None = None) -> EncoderOutput:
        self._validate_batch(categorical, continuous, lengths)
        event = self._event_tensor(categorical, continuous, augment=augment)
        sequence = self._transformer_sequence(event, lengths)
        slow_weights = self._timescale_weights(
            lengths, sequence.shape[1], self.persistent_half_life_events,
            device=sequence.device, dtype=sequence.dtype,
        )
        fast_weights = self._timescale_weights(
            lengths, sequence.shape[1], self.context_half_life_events,
            device=sequence.device, dtype=sequence.dtype,
        )
        slow_state = (sequence * slow_weights.unsqueeze(-1)).sum(dim=1)
        fast_state = (sequence * fast_weights.unsqueeze(-1)).sum(dim=1)
        persistent = self.output_projection(slow_state)
        context = self.context_projection(fast_state - slow_state)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((persistent, context), dim=1)))
        combined = self.fusion_normalization(persistent + gate * context)
        if self.ablation == "persistent_only":
            context, combined = torch.zeros_like(context), persistent
        elif self.ablation == "context_only":
            persistent, combined = torch.zeros_like(persistent), context
        return EncoderOutput(persistent=persistent, context=context, combined=combined)


ModelFactory = Callable[
    [dict[str, dict[str, int]], int, dict[str, Any], list[str] | None],
    SingleVectorEncoder,
]
LEGACY_SINGLE_VECTOR_VARIANT = "single_vector"
MODEL_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register an observed-input-only model factory under a stable name."""
    if not name or name in MODEL_REGISTRY:
        raise ValueError(f"Invalid or duplicate model variant: {name!r}")

    def decorator(factory: ModelFactory) -> ModelFactory:
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def configured_model_variant(config: dict[str, Any]) -> str:
    variant = str(config.get("model", {}).get("variant", LEGACY_SINGLE_VECTOR_VARIANT))
    if variant not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model variant {variant!r}; available variants: {sorted(MODEL_REGISTRY)}"
        )
    return variant


@register_model(LEGACY_SINGLE_VECTOR_VARIANT)
def _build_single_vector(
    vocabularies: dict[str, dict[str, int]],
    continuous_dim: int,
    config: dict[str, Any],
    categorical_fields: list[str] | None = None,
) -> SingleVectorEncoder:
    return SingleVectorEncoder(
        vocabularies,
        continuous_dim,
        config,
        categorical_fields=categorical_fields,
    )


@register_model("factorized_pc")
def _build_factorized_pc(vocabularies: dict[str, dict[str, int]], continuous_dim: int,
                         config: dict[str, Any], categorical_fields: list[str] | None = None
                         ) -> SingleVectorEncoder:
    return FactorizedPCEncoder(vocabularies, continuous_dim, config, categorical_fields)


@register_model("two_timescale_pc")
def _build_two_timescale_pc(vocabularies: dict[str, dict[str, int]], continuous_dim: int,
                            config: dict[str, Any], categorical_fields: list[str] | None = None
                            ) -> SingleVectorEncoder:
    return TwoTimescalePCEncoder(vocabularies, continuous_dim, config, categorical_fields)


@register_model("slow_fast_v1")
def _build_slow_fast_v1(vocabularies: dict[str, dict[str, int]], continuous_dim: int,
                        config: dict[str, Any], categorical_fields: list[str] | None = None
                        ) -> SingleVectorEncoder:
    return SlowFastV1Encoder(vocabularies, continuous_dim, config, categorical_fields)


@register_model("causal_transformer_pc")
def _build_causal_transformer_pc(
    vocabularies: dict[str, dict[str, int]],
    continuous_dim: int,
    config: dict[str, Any],
    categorical_fields: list[str] | None = None,
) -> SingleVectorEncoder:
    return CausalTransformerPCEncoder(
        vocabularies, continuous_dim, config, categorical_fields
    )


def _capacity_matched_single(
    target_factory: type[SingleVectorEncoder],
    target_variant: str,
    vocabularies: dict[str, dict[str, int]],
    continuous_dim: int,
    config: dict[str, Any],
    categorical_fields: list[str] | None,
) -> SingleVectorEncoder:
    """Build a single-GRU control closest to one candidate's parameter budget."""
    target_config = {
        **config,
        "model": {
            **config["model"],
            "variant": target_variant,
            "user_embedding_dim": int(config["model"].get("matched_output_dim", 128)),
        },
    }
    target = target_factory(vocabularies, continuous_dim, target_config, categorical_fields)
    target_count = sum(parameter.numel() for parameter in target.parameters())
    del target
    output_dim = int(target_config["model"]["user_embedding_dim"])
    low, high = 1, max(2, int(config["model"].get("capacity_search_max_hidden_dim", 512)))
    best: tuple[int, SingleVectorEncoder] | None = None
    while low <= high:
        hidden = (low + high) // 2
        candidate_config = {
            **config,
            "model": {
                **config["model"],
                "variant": "single_vector",
                "hidden_dim": hidden,
                "user_embedding_dim": output_dim,
            },
        }
        candidate = SingleVectorEncoder(
            vocabularies, continuous_dim, candidate_config, categorical_fields
        )
        count = sum(parameter.numel() for parameter in candidate.parameters())
        if best is None or abs(count - target_count) < abs(best[0] - target_count):
            best = (count, candidate)
        if count < target_count:
            low = hidden + 1
        else:
            high = hidden - 1
    assert best is not None
    best[1].capacity_match = {
        "target_model_variant": target_variant,
        "target_parameters": target_count,
        "actual_parameters": best[0],
        "relative_error": abs(best[0] - target_count) / target_count,
    }
    return best[1]


@register_model("capacity_matched_single")
def _build_capacity_matched_single(vocabularies: dict[str, dict[str, int]], continuous_dim: int,
                                   config: dict[str, Any], categorical_fields: list[str] | None = None
                                   ) -> SingleVectorEncoder:
    return _capacity_matched_single(
        FactorizedPCEncoder,
        "factorized_pc",
        vocabularies,
        continuous_dim,
        config,
        categorical_fields,
    )


@register_model("two_timescale_capacity_matched_single")
def _build_two_timescale_capacity_matched_single(
    vocabularies: dict[str, dict[str, int]],
    continuous_dim: int,
    config: dict[str, Any],
    categorical_fields: list[str] | None = None,
) -> SingleVectorEncoder:
    return _capacity_matched_single(
        TwoTimescalePCEncoder,
        "two_timescale_pc",
        vocabularies,
        continuous_dim,
        config,
        categorical_fields,
    )


@register_model("slow_fast_capacity_matched_single")
def _build_slow_fast_capacity_matched_single(
    vocabularies: dict[str, dict[str, int]], continuous_dim: int,
    config: dict[str, Any], categorical_fields: list[str] | None = None,
) -> SingleVectorEncoder:
    return _capacity_matched_single(
        SlowFastV1Encoder,
        "slow_fast_v1",
        vocabularies,
        continuous_dim,
        config,
        categorical_fields,
    )


@register_model("causal_transformer_capacity_matched_single")
def _build_causal_transformer_capacity_matched_single(
    vocabularies: dict[str, dict[str, int]],
    continuous_dim: int,
    config: dict[str, Any],
    categorical_fields: list[str] | None = None,
) -> SingleVectorEncoder:
    return _capacity_matched_single(
        CausalTransformerPCEncoder,
        "causal_transformer_pc",
        vocabularies,
        continuous_dim,
        config,
        categorical_fields,
    )


# Stable names let ablation experiments differ in one configuration field while
# retaining the same architecture and artifact contract.
for _variant, _ablation in (("persistent_only", "persistent_only"),
                            ("context_only", "context_only")):
    def _factory(vocabularies: dict[str, dict[str, int]], continuous_dim: int,
                 config: dict[str, Any], categorical_fields: list[str] | None = None,
                 *, ablation: str = _ablation) -> SingleVectorEncoder:
        copied = {**config, "model": {**config["model"], "ablation": ablation}}
        return FactorizedPCEncoder(vocabularies, continuous_dim, copied, categorical_fields)
    register_model(_variant)(_factory)


def build_model(
    vocabularies: dict[str, dict[str, int]],
    continuous_dim: int,
    config: dict[str, Any],
    categorical_fields: list[str] | None = None,
) -> SingleVectorEncoder:
    """Construct the configured model from public prepared-data contracts only."""
    variant = configured_model_variant(config)
    return MODEL_REGISTRY[variant](
        vocabularies, continuous_dim, config, categorical_fields
    )
