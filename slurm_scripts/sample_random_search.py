#!/usr/bin/env python3

import argparse
import csv
import random
from math import prod
from pathlib import Path

from omegaconf import OmegaConf


METADATA_KEYS = {"SEED"}


def read_existing(manifest_path: Path) -> set[tuple[str, ...]]:
    existing = set()
    if not manifest_path.exists():
        return existing

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            existing.add(freeze_fields(row))
    return existing


def next_combo_index(manifest_path: Path) -> int:
    if not manifest_path.exists():
        return 1

    max_idx = 0
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            label = row[0]
            if not label.startswith("C"):
                continue
            combo_idx = label[1:].split("_", 1)[0]
            if combo_idx.isdigit():
                max_idx = max(max_idx, int(combo_idx))
    return max_idx + 1


def normalize_scalar(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "null"
    return str(value)


def flatten_override(key: str, value) -> list[str]:
    if isinstance(value, dict):
        flattened = []
        for child_key, child_value in value.items():
            flattened.extend(flatten_override(f"{key}.{child_key}", child_value))
        return flattened
    return [f"{key}={normalize_scalar(value)}"]


def parse_parameter_specs(parameters: dict) -> tuple[dict[str, list], dict[str, object]]:
    search_params = {}
    fixed_params = {}

    for key, spec in parameters.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Expected parameter spec for {key} to be a mapping.")
        if "values" in spec:
            values = spec["values"]
            if not isinstance(values, list) or not values:
                raise ValueError(f"Expected {key}.values to be a non-empty list.")
            search_params[key] = values
        elif "value" in spec:
            if isinstance(spec["value"], list):
                raise ValueError(f"{key}.value is a list; did you mean 'values'?")
            fixed_params[key] = spec["value"]
        else:
            raise ValueError(f"Expected {key} to define either 'value' or 'values'.")

    return search_params, fixed_params


def resolve_seeds(sweep_cfg) -> list[str]:
    sweep_seeds_cfg = sweep_cfg.get("SWEEP_SEEDS")
    if sweep_seeds_cfg is not None:
        sweep_seeds = OmegaConf.to_container(sweep_seeds_cfg, resolve=True)
        if not isinstance(sweep_seeds, list) or not sweep_seeds:
            raise ValueError("Expected SWEEP_SEEDS to be a non-empty list.")
        return [normalize_scalar(seed) for seed in sweep_seeds]

    return []


def print_search_space(
    search_params: dict[str, list], fixed_params: dict[str, object], seeds: list[str]
) -> None:
    print(
        f"Search space ({len(search_params)} parameters searched, "
        f"{len(fixed_params)} fixed):"
    )
    key_width = max((len(key) for key in search_params), default=0)
    for key, values in search_params.items():
        print(f"  {key:<{key_width}} : {values}")
    if fixed_params:
        print(f"Fixed: {', '.join(fixed_params)}")
    if search_params:
        sizes = " x ".join(str(len(values)) for values in search_params.values())
        total = prod(len(values) for values in search_params.values())
        print(f"Total: {sizes} = {total} combinations")
    else:
        print("Total: 1 combination (no searched parameters)")
    if seeds:
        print(f"Seeds per combo: [{', '.join(seeds)}] -> {len(seeds)} runs per combination")
    else:
        print("Seeds per combo: none (one run per combination, SEED from the base config)")


def is_metadata_field(field: str) -> bool:
    if "=" not in field:
        return True
    return field.split("=", 1)[0] in METADATA_KEYS


def freeze_fields(fields: list[str]) -> tuple[str, ...]:
    combo_fields = [
        field for field in fields if field and "=" in field and not is_metadata_field(field)
    ]
    return tuple(sorted(combo_fields))


def format_combo_fields(combo: dict[str, object], fixed_params: dict[str, object]) -> list[str]:
    fields = []
    for key, value in fixed_params.items():
        fields.extend(flatten_override(key, value))
    for key, value in combo.items():
        fields.extend(flatten_override(key, value))
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1.")

    sweep_cfg = OmegaConf.load(Path(args.sweep_config))
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    parameters_cfg = sweep_cfg.get("parameters")
    parameters = OmegaConf.to_container(parameters_cfg, resolve=True)
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Expected sweep config to contain a non-empty 'parameters' mapping.")

    search_params, fixed_params = parse_parameter_specs(parameters)
    seeds = resolve_seeds(sweep_cfg)
    print_search_space(search_params, fixed_params, seeds)

    total_combos = prod(len(values) for values in search_params.values()) if search_params else 1

    rng = random.Random(args.random_seed)
    existing = read_existing(output_file)
    combo_idx = next_combo_index(output_file)
    created = []
    created_combos = 0

    remaining = max(total_combos - len(existing), 0)
    target_samples = min(args.num_samples, remaining)
    max_attempts = max(1000, max(target_samples, 1) * 100)
    attempts = 0

    while created_combos < target_samples:
        combo = {key: rng.choice(values) for key, values in search_params.items()}
        fields = format_combo_fields(combo, fixed_params)
        frozen = freeze_fields(fields)

        if frozen in existing:
            attempts += 1
            if attempts >= max_attempts:
                raise RuntimeError("Failed to sample enough unique combinations.")
            continue

        existing.add(frozen)
        for seed_id, seed in enumerate(seeds, start=1):
            label = f"C{combo_idx:04d}_{seed_id}" if len(seeds) > 1 else f"C{combo_idx:04d}"
            created.append([label, f"SEED={seed}", *fields])
        if not seeds:
            created.append([f"C{combo_idx:04d}", *fields])
        combo_idx += 1
        created_combos += 1

    if created:
        with output_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(created)

    print(f"Sampled so far (excluding seeds): {len(existing)} / {total_combos} combinations")


if __name__ == "__main__":
    main()
