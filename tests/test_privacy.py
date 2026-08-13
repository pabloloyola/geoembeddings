from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from geoembeddings.privacy_evaluation import PRIVACY_CONFIG_SCHEMA, load_privacy_config, validate_privacy_config


CONFIG = Path("configs/privacy/diagnostic_v1.yaml")


def raw_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_diagnostic_privacy_config_is_typed_and_complete() -> None:
    config = load_privacy_config(CONFIG)
    assert config.schema_version == PRIVACY_CONFIG_SCHEMA
    assert config.split.train_fraction + config.split.validation_fraction + config.split.test_fraction == 1
    assert config.membership["statistical_baseline"]["training_membership_status"] == "not_applicable"
    assert config.sensitive_attributes[0].derivation.version.endswith("/1.0")


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c.update({"surprise": True}), "unknown fields"),
    (lambda c: c["splits"].update({"test_fraction": 0.3}), "sum to 1"),
    (lambda c: c["matching"]["variables"][0].update({"bin_boundaries": [0, 10, 10]}), "overlapping"),
    (lambda c: c["support"].update({"minimum_total": 0}), "positive"),
    (lambda c: c["sensitive_attributes"][0]["derivation"].update({"version": "latest"}), "versioned"),
    (lambda c: c["membership"].update({"nonmember_definition": c["membership"]["member_definition"]}), "overlap"),
])
def test_privacy_config_rejects_invalid_protocol(mutation, match: str) -> None:
    config = deepcopy(raw_config())
    mutation(config)
    with pytest.raises(ValueError, match=match):
        validate_privacy_config(config)


def test_privacy_config_rejects_unfrozen_derivation() -> None:
    config = raw_config()
    config["sensitive_attributes"][0]["derivation"]["method"] = "quantiles_fit_on_all_labels"
    with pytest.raises(ValueError, match="not frozen"):
        validate_privacy_config(config)
