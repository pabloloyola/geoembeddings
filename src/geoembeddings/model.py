from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .data import TARGET_FIELDS


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
        for objective, field in TARGET_FIELDS.items():
            if objective in config["objectives"] and field in vocabularies:
                self.heads[objective] = nn.Linear(output_dim, len(vocabularies[field]))

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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedding = self.encode(categorical, continuous, lengths, augment=True)
        logits = {name: head(embedding) for name, head in self.heads.items()}
        return embedding, logits

    def encode_components(
        self,
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        lengths: torch.Tensor,
        augment: bool = False,
    ) -> EncoderOutput:
        """Return the legacy representation through the named output boundary."""
        return self.output_adapter(
            self.encode(categorical, continuous, lengths, augment=augment)
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
                          lengths: torch.Tensor, augment: bool = False) -> EncoderOutput:
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
                lengths: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
                          lengths: torch.Tensor, augment: bool = False) -> EncoderOutput:
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
                lengths: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.encode_components(categorical, continuous, lengths, augment=True)
        logits = {
            name: head(getattr(output, self.loss_routes[name]))
            for name, head in self.heads.items()
        }
        return output.combined, logits


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
