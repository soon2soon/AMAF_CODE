#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from .validation import (
    DEFAULT_CONTRACT,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INVENTORY = (
    ROOT
    / "analysis"
    / "figures"
    / "figure_inventory.yaml"
)


def load_inventory(
    path: Path = DEFAULT_INVENTORY,
):

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(
            "Inventory root must be a mapping"
        )

    if "figures" not in data:
        raise KeyError(
            "Inventory missing figures"
        )

    return data


def validate_inventory(
    inventory,
    contract,
):
    errors = []
    warnings = []

    vocab = inventory.get(
        "vocabulary",
        {},
    )

    allowed_status = set(
        vocab.get("statuses", [])
    )

    allowed_roles = set(
        vocab.get("roles", [])
    )

    allowed_priorities = set(
        vocab.get("priorities", [])
    )

    allowed_readiness = set(
        vocab.get("readiness", [])
    )

    figures = inventory.get(
        "figures",
        {},
    )

    contract_figures = contract.get(
        "figures",
        {},
    )

    seen_ids = set()

    for key, spec in figures.items():

        loc = f"figures.{key}"

        required = [
            "figure_id",
            "domain",
            "role",
            "status",
            "priority",
            "readiness",
            "contract_key",
            "requires_new_experiment",
        ]

        for field in required:
            if field not in spec:
                errors.append(
                    f"{loc}: missing {field}"
                )

        figure_id = spec.get(
            "figure_id"
        )

        if figure_id in seen_ids:
            errors.append(
                f"{loc}: duplicate figure_id {figure_id}"
            )

        seen_ids.add(
            figure_id
        )

        status = spec.get(
            "status"
        )

        role = spec.get(
            "role"
        )

        priority = spec.get(
            "priority"
        )

        readiness = spec.get(
            "readiness"
        )

        if (
            allowed_status
            and status not in allowed_status
        ):
            errors.append(
                f"{loc}: unknown status {status}"
            )

        if (
            allowed_roles
            and role not in allowed_roles
        ):
            errors.append(
                f"{loc}: unknown role {role}"
            )

        if (
            allowed_priorities
            and priority not in allowed_priorities
        ):
            errors.append(
                f"{loc}: unknown priority {priority}"
            )

        if (
            allowed_readiness
            and readiness not in allowed_readiness
        ):
            errors.append(
                f"{loc}: unknown readiness {readiness}"
            )

        contract_key = spec.get(
            "contract_key"
        )

        if contract_key is not None:

            if contract_key not in contract_figures:
                errors.append(
                    (
                        f"{loc}: contract_key "
                        f"{contract_key} not found"
                    )
                )

        if status == "active":

            if contract_key is None:
                errors.append(
                    (
                        f"{loc}: active figure must "
                        "have contract_key"
                    )
                )

        if (
            readiness == "blocked"
            and status != "blocked"
        ):
            warnings.append(
                (
                    f"{loc}: readiness=blocked but "
                    f"status={status}"
                )
            )

    return errors, warnings


def print_summary(
    inventory,
):

    figures = inventory[
        "figures"
    ]

    print(
        "=== AMAF FIGURE INVENTORY ==="
    )

    print(
        "version:",
        inventory.get(
            "inventory_version"
        ),
    )

    print(
        "total figures:",
        len(figures),
    )

    print()

    for field in [
        "status",
        "role",
        "domain",
        "priority",
        "readiness",
    ]:

        counter = Counter(
            spec.get(
                field,
                "-"
            )
            for spec in figures.values()
        )

        print(
            field.upper()
        )

        for key, n in sorted(
            counter.items(),
            key=lambda x: (
                str(x[0])
            ),
        ):
            print(
                f"  {str(key):24s} {n:3d}"
            )

        print()


def print_list(
    inventory,
    *,
    status=None,
    role=None,
    domain=None,
    priority=None,
):

    figures = inventory[
        "figures"
    ]

    rows = []

    for key, spec in figures.items():

        if (
            status is not None
            and spec.get("status") != status
        ):
            continue

        if (
            role is not None
            and spec.get("role") != role
        ):
            continue

        if (
            domain is not None
            and spec.get("domain") != domain
        ):
            continue

        if (
            priority is not None
            and spec.get("priority") != priority
        ):
            continue

        rows.append(
            (
                spec.get("priority", "-"),
                spec.get("status", "-"),
                spec.get("domain", "-"),
                spec.get("role", "-"),
                key,
            )
        )

    rows.sort()

    print(
        f"{'PRI':4s} "
        f"{'STATUS':12s} "
        f"{'DOMAIN':14s} "
        f"{'ROLE':18s} "
        f"KEY"
    )

    print(
        "-" * 90
    )

    for (
        pri,
        stat,
        dom,
        role_,
        key,
    ) in rows:

        print(
            f"{pri:4s} "
            f"{stat:12s} "
            f"{dom:14s} "
            f"{role_:18s} "
            f"{key}"
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect and validate AMAF "
            "figure inventory"
        )
    )

    parser.add_argument(
        "--summary",
        action="store_true",
    )

    parser.add_argument(
        "--list",
        action="store_true",
    )

    parser.add_argument(
        "--status",
    )

    parser.add_argument(
        "--role",
    )

    parser.add_argument(
        "--domain",
    )

    parser.add_argument(
        "--priority",
    )

    args = parser.parse_args()

    inventory = load_inventory()

    contract = load_contract(
        DEFAULT_CONTRACT
    )

    errors, warnings = (
        validate_inventory(
            inventory,
            contract,
        )
    )

    for msg in warnings:
        print(
            "[WARNING]",
            msg,
        )

    for msg in errors:
        print(
            "[ERROR]",
            msg,
        )

    if errors:

        print()
        print(
            "FIGURE INVENTORY: FAIL"
        )

        return 1

    print(
        "FIGURE INVENTORY: PASS"
    )

    print()

    if (
        args.summary
        or not args.list
    ):
        print_summary(
            inventory
        )

    if args.list:

        print_list(
            inventory,
            status=args.status,
            role=args.role,
            domain=args.domain,
            priority=args.priority,
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
