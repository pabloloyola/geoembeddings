from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .data import TARGET_FIELDS


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
        for objective, field in TARGET_FIELDS.items():
            if objective in config["objectives"] and field in vocabularies:
                self.heads[objective] = nn.Linear(output_dim, len(vocabularies[field]))

    def encode(
        self,
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        lengths: torch.Tensor,
        augment: bool = False,
    ) -> torch.Tensor:
        self._validate_batch(categorical, continuous, lengths)
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
        time_steps = sequence_output.shape[1]
        cpu_positions = torch.arange(time_steps, device="cpu").unsqueeze(0)
        cpu_final_positions = (lengths.detach().cpu() - 1).unsqueeze(1)
        final_mask = (cpu_positions == cpu_final_positions).to(
            device=sequence_output.device,
            dtype=sequence_output.dtype,
        )
        final_hidden = (sequence_output * final_mask.unsqueeze(-1)).sum(dim=1)
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


def build_model(
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
