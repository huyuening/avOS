"""Helpers for selecting trajectory reconstruction implementations."""

import importlib


DEFAULT_RECONSTRUCTOR = "motion_reconstructor_v2"

RECONSTRUCTOR_ALIASES = {
    "2": "motion_reconstructor_v2",
    "v2": "motion_reconstructor_v2",
    "new": "motion_reconstructor_v2",
    "velocity_heading": "motion_reconstructor_v2",
    "velocity_heading_reconstructor": "motion_reconstructor_v2",
    "motion_reconstructor_v2": "motion_reconstructor_v2",
}


def normalize_reconstructor_method(method=None):
    if method is None:
        return DEFAULT_RECONSTRUCTOR
    key = str(method).strip()
    if not key:
        return DEFAULT_RECONSTRUCTOR
    key = key.replace("-", "_")
    key = key.split(".")[-1]
    normalized = RECONSTRUCTOR_ALIASES.get(key)
    if normalized is None:
        valid = ", ".join(sorted(set(RECONSTRUCTOR_ALIASES.values())))
        raise ValueError(f"Unsupported reconstructor '{method}'. Use one of: {valid}.")
    return normalized


def load_reconstructor_module(method=None):
    module_name = normalize_reconstructor_method(method)
    primary_name = f"reconstructor.{module_name}"
    try:
        return importlib.import_module(primary_name)
    except ModuleNotFoundError as exc:
        if exc.name not in ("reconstructor", primary_name):
            raise
        return importlib.import_module(f"d2s.reconstructor.{module_name}")


def get_reconstructor_main(method=None):
    return load_reconstructor_module(method).main


def get_object_track_reconstruction(method=None):
    return load_reconstructor_module(method).object_track_reconstruction
