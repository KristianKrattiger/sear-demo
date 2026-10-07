from pathlib import Path

import yaml


def load_config(path: str | Path = "config/default.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `section.key=value` overrides in place (values parsed as YAML), so an eval can run
    a flag on without editing the config file. An unknown section or key is an error, not a new
    setting: a typo must not silently run the default."""
    for item in overrides:
        path, sep, raw = item.partition("=")
        keys = path.strip().split(".")
        if not sep or len(keys) < 2:
            raise ValueError(f"override {item!r} is not section.key=value")
        node = cfg
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                raise KeyError(f"override {item!r}: no config section {k!r}")
            node = node[k]
        if keys[-1] not in node:
            raise KeyError(f"override {item!r}: no config key {keys[-1]!r} in {'.'.join(keys[:-1])}")
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg
