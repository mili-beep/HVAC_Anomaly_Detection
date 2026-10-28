import pytest

from ahu_fdd.feature_sensitivity import resolve_feature_variants


def test_feature_variants_remove_direct_state_and_pump_history() -> None:
    manifest = {
        "feature_variants": {
            "physical_signal": [
                "sensor__supply_air_temperature",
                "sensor__supply_fan",
                "sensor__heating_pump_1",
                "relationship__supply_return_delta",
                "relationship__fan_valve_interaction",
                "temporal__past_mean_3__supply_air_temperature",
                "temporal__past_mean_3__supply_fan",
            ]
        }
    }
    config = {
        "default_bms_signals": {
            "source_variant": "physical_signal",
            "expected_source_feature_count": 7,
            "excluded_sensor_bases": [],
            "excluded_feature_names": [],
        },
        "symptom_only": {
            "source_variant": "physical_signal",
            "expected_source_feature_count": 7,
            "excluded_sensor_bases": ["supply_fan", "heating_pump_1"],
            "excluded_feature_names": ["relationship__fan_valve_interaction"],
        },
    }
    variants = resolve_feature_variants(manifest, config)
    assert len(variants["default_bms_signals"]) == 7
    assert variants["symptom_only"] == [
        "sensor__supply_air_temperature",
        "relationship__supply_return_delta",
        "temporal__past_mean_3__supply_air_temperature",
    ]


def test_feature_variants_reject_stale_source_count() -> None:
    manifest = {"feature_variants": {"physical_signal": ["sensor__x"]}}
    config = {
        "default_bms_signals": {
            "source_variant": "physical_signal",
            "expected_source_feature_count": 88,
        },
        "symptom_only": {
            "source_variant": "physical_signal",
            "expected_source_feature_count": 88,
        },
    }
    with pytest.raises(ValueError, match="Frozen source feature count"):
        resolve_feature_variants(manifest, config)
