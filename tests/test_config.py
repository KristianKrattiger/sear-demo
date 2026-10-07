import pytest

from sear_demo.config import apply_overrides


def test_overrides_set_existing_keys_and_parse_values():
    cfg = {"verifier": {"tau": 0.5, "granularity": "claim"}, "gate": {"partial": False}}
    apply_overrides(cfg, ["verifier.granularity=sentence", "verifier.tau=0.7", "gate.partial=true"])
    assert cfg == {"verifier": {"tau": 0.7, "granularity": "sentence"}, "gate": {"partial": True}}


@pytest.mark.parametrize("bad", ["verifier.taux=1", "verfier.tau=1", "tau=1", "verifier.tau"])
def test_an_override_typo_is_an_error_not_a_new_setting(bad):
    cfg = {"verifier": {"tau": 0.5}}
    with pytest.raises((KeyError, ValueError)):
        apply_overrides(cfg, [bad])
    assert cfg == {"verifier": {"tau": 0.5}}
