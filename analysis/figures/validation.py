#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONTRACT = (
    ROOT
    / "analysis"
    / "figures"
    / "figure_contract.yaml"
)


# ======================================================================
# Validation messages
# ======================================================================

@dataclass
class ValidationMessage:
    level: str
    location: str
    message: str


# ======================================================================
# Contract validator
# ======================================================================

class ContractValidator:

    def __init__(
        self,
        contract: dict[str, Any],
        root: Path,
        check_files: bool = True,
        check_builders: bool = False,
    ):
        self.contract = contract
        self.root = root
        self.check_files = check_files
        self.check_builders = check_builders
        self.messages: list[ValidationMessage] = []

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def error(
        self,
        location: str,
        message: str,
    ):
        self.messages.append(
            ValidationMessage(
                level="ERROR",
                location=location,
                message=message,
            )
        )

    def warning(
        self,
        location: str,
        message: str,
    ):
        self.messages.append(
            ValidationMessage(
                level="WARNING",
                location=location,
                message=message,
            )
        )

    def info(
        self,
        location: str,
        message: str,
    ):
        self.messages.append(
            ValidationMessage(
                level="INFO",
                location=location,
                message=message,
            )
        )

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_string_set(values) -> set[str]:
        return {
            str(v).strip()
            for v in values
        }

    def _allowed(
        self,
        key: str,
    ) -> set[str]:

        vocab = self.contract.get(
            "vocabulary",
            {},
        )

        values = vocab.get(
            key,
            [],
        )

        return {
            str(x)
            for x in values
        }

    def _dataset_lifecycle(
        self,
        spec: dict[str, Any],
    ) -> str:

        explicit = spec.get(
            "lifecycle"
        )

        if explicit:
            return str(explicit)

        kind = spec.get(
            "kind"
        )

        defaults = {
            "canonical_csv": "canonical",
            "analysis_csv": "canonical",
            "frozen_dir": "frozen",
            "live_output": "live",
        }

        return defaults.get(
            kind,
            "canonical",
        )

    # ------------------------------------------------------------------
    # Top-level validation
    # ------------------------------------------------------------------

    def validate_top_level(self):

        required = [
            "schema_version",
            "contract_version",
            "vocabulary",
            "global_policy",
            "metrics",
            "datasets",
            "figures",
        ]

        for key in required:

            if key not in self.contract:
                self.error(
                    "root",
                    (
                        "Missing required "
                        f"top-level key: {key}"
                    ),
                )

        if not isinstance(
            self.contract.get(
                "metrics",
                {},
            ),
            dict,
        ):
            self.error(
                "metrics",
                "metrics must be a mapping",
            )

        if not isinstance(
            self.contract.get(
                "datasets",
                {},
            ),
            dict,
        ):
            self.error(
                "datasets",
                "datasets must be a mapping",
            )

        if not isinstance(
            self.contract.get(
                "figures",
                {},
            ),
            dict,
        ):
            self.error(
                "figures",
                "figures must be a mapping",
            )

    # ------------------------------------------------------------------
    # Metric validation
    # ------------------------------------------------------------------

    def validate_metrics(self):

        metrics = self.contract.get(
            "metrics",
            {},
        )

        allowed_directions = self._allowed(
            "directions"
        )

        for metric_id, spec in metrics.items():

            loc = (
                f"metrics.{metric_id}"
            )

            if not isinstance(
                spec,
                dict,
            ):
                self.error(
                    loc,
                    "Metric spec must be a mapping",
                )
                continue

            for required in [
                "domain",
                "unit",
                "direction",
                "definition",
            ]:
                if required not in spec:
                    self.error(
                        loc,
                        (
                            "Missing metric field: "
                            f"{required}"
                        ),
                    )

            direction = spec.get(
                "direction"
            )

            if (
                direction is not None
                and allowed_directions
                and direction
                not in allowed_directions
            ):
                self.error(
                    loc,
                    (
                        "Unknown direction: "
                        f"{direction}"
                    ),
                )

    # ------------------------------------------------------------------
    # Dataset validation
    # ------------------------------------------------------------------

    def validate_datasets(self):

        datasets = self.contract.get(
            "datasets",
            {},
        )

        allowed_kinds = self._allowed(
            "dataset_kinds"
        )

        allowed_lifecycles = self._allowed(
            "dataset_lifecycles"
        )

        for dataset_id, spec in datasets.items():

            loc = (
                f"datasets.{dataset_id}"
            )

            if not isinstance(
                spec,
                dict,
            ):
                self.error(
                    loc,
                    "Dataset spec must be a mapping",
                )
                continue

            for required in [
                "path",
                "kind",
                "manuscript_allowed",
            ]:
                if required not in spec:
                    self.error(
                        loc,
                        (
                            "Missing dataset field: "
                            f"{required}"
                        ),
                    )

            kind = spec.get(
                "kind"
            )

            if (
                kind is not None
                and allowed_kinds
                and kind
                not in allowed_kinds
            ):
                self.error(
                    loc,
                    (
                        "Unknown dataset kind: "
                        f"{kind}"
                    ),
                )

            lifecycle = (
                self._dataset_lifecycle(
                    spec
                )
            )

            if (
                allowed_lifecycles
                and lifecycle
                not in allowed_lifecycles
            ):
                self.error(
                    loc,
                    (
                        "Unknown dataset lifecycle: "
                        f"{lifecycle}"
                    ),
                )

            if not self.check_files:
                continue

            raw_path = spec.get(
                "path"
            )

            if not raw_path:
                continue

            path = (
                self.root
                / raw_path
            )

            optional = bool(
                spec.get(
                    "optional",
                    False,
                )
            )

            if not path.exists():

                nonfatal_missing = {
                    "planned",
                    "live",
                    "retired",
                }

                if (
                    optional
                    or lifecycle
                    in nonfatal_missing
                ):
                    self.warning(
                        loc,
                        (
                            f"{lifecycle} dataset "
                            f"not present: {path}"
                        ),
                    )

                else:
                    self.error(
                        loc,
                        (
                            f"Required {lifecycle} "
                            "dataset path missing: "
                            f"{path}"
                        ),
                    )

                continue

            if kind in {
                "canonical_csv",
                "analysis_csv",
            }:
                self.validate_csv_dataset(
                    dataset_id=dataset_id,
                    spec=spec,
                    path=path,
                )

    def validate_csv_dataset(
        self,
        dataset_id: str,
        spec: dict[str, Any],
        path: Path,
    ):

        loc = (
            f"datasets.{dataset_id}"
        )

        try:
            with path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as f:

                reader = csv.DictReader(
                    f
                )

                fieldnames = (
                    reader.fieldnames
                    or []
                )

                rows = list(
                    reader
                )

        except Exception as exc:
            self.error(
                loc,
                (
                    "Could not read CSV: "
                    f"{exc}"
                ),
            )
            return

        required_columns = spec.get(
            "required_columns",
            [],
        )

        missing = [
            col
            for col in required_columns
            if col not in fieldnames
        ]

        if missing:
            self.error(
                loc,
                (
                    "Missing required CSV "
                    "columns: "
                    + ", ".join(missing)
                ),
            )

        expected_rows = spec.get(
            "expected_rows"
        )

        if (
            expected_rows is not None
            and len(rows)
            != int(expected_rows)
        ):
            self.error(
                loc,
                (
                    f"Expected {expected_rows} "
                    f"rows, found {len(rows)}"
                ),
            )

        expected_unique = spec.get(
            "expected_unique",
            {},
        )

        for column, expected in (
            expected_unique.items()
        ):

            if column not in fieldnames:
                self.error(
                    loc,
                    (
                        "Cannot validate "
                        "expected_unique; "
                        f"missing column {column}"
                    ),
                )
                continue

            observed = (
                self._as_string_set(
                    row[column]
                    for row in rows
                )
            )

            expected_set = (
                self._as_string_set(
                    expected
                )
            )

            if observed != expected_set:

                missing_values = (
                    expected_set
                    - observed
                )

                extra_values = (
                    observed
                    - expected_set
                )

                detail = []

                if missing_values:
                    detail.append(
                        "missing="
                        + str(
                            sorted(
                                missing_values
                            )
                        )
                    )

                if extra_values:
                    detail.append(
                        "extra="
                        + str(
                            sorted(
                                extra_values
                            )
                        )
                    )

                self.error(
                    loc,
                    (
                        "Unexpected values in "
                        f"{column}: "
                        + "; ".join(detail)
                    ),
                )

    # ------------------------------------------------------------------
    # Figure validation
    # ------------------------------------------------------------------

    def validate_figures(self):

        figures = self.contract.get(
            "figures",
            {},
        )

        metrics = self.contract.get(
            "metrics",
            {},
        )

        datasets = self.contract.get(
            "datasets",
            {},
        )

        allowed_groups = self._allowed(
            "figure_groups"
        )

        allowed_statuses = self._allowed(
            "statuses"
        )

        allowed_pairings = self._allowed(
            "pairings"
        )

        allowed_uncertainty = self._allowed(
            "uncertainty_methods"
        )

        figure_ids: set[str] = set()

        for key, spec in figures.items():

            loc = (
                f"figures.{key}"
            )

            if not isinstance(
                spec,
                dict,
            ):
                self.error(
                    loc,
                    "Figure spec must be a mapping",
                )
                continue

            for required in [
                "figure_id",
                "group",
                "status",
                "domain",
                "builder",
                "experimental_unit",
                "panels",
            ]:
                if required not in spec:
                    self.error(
                        loc,
                        (
                            "Missing figure field: "
                            f"{required}"
                        ),
                    )

            figure_id = spec.get(
                "figure_id"
            )

            if figure_id:

                figure_id = str(
                    figure_id
                )

                if figure_id in figure_ids:
                    self.error(
                        loc,
                        (
                            "Duplicate figure_id: "
                            f"{figure_id}"
                        ),
                    )

                figure_ids.add(
                    figure_id
                )

            group = spec.get(
                "group"
            )

            if (
                group is not None
                and allowed_groups
                and group
                not in allowed_groups
            ):
                self.error(
                    loc,
                    (
                        "Unknown figure group: "
                        f"{group}"
                    ),
                )

            status = spec.get(
                "status"
            )

            if (
                status is not None
                and allowed_statuses
                and status
                not in allowed_statuses
            ):
                self.error(
                    loc,
                    (
                        "Unknown figure status: "
                        f"{status}"
                    ),
                )

            builder = spec.get(
                "builder"
            )

            if builder:
                self.validate_builder_string(
                    location=loc,
                    builder=builder,
                )

                if self.check_builders:
                    self.validate_builder_import(
                        location=loc,
                        builder=builder,
                    )

            panels = spec.get(
                "panels",
                {},
            )

            if not isinstance(
                panels,
                dict,
            ):
                self.error(
                    loc,
                    "panels must be a mapping",
                )
                continue

            if not panels:
                self.warning(
                    loc,
                    "Figure contains no panels",
                )

            for panel_id, panel in (
                panels.items()
            ):

                ploc = (
                    f"{loc}.panels."
                    f"{panel_id}"
                )

                if not isinstance(
                    panel,
                    dict,
                ):
                    self.error(
                        ploc,
                        "Panel must be a mapping",
                    )
                    continue

                panel_status = panel.get(
                    "status"
                )

                if (
                    panel_status is not None
                    and allowed_statuses
                    and panel_status
                    not in allowed_statuses
                ):
                    self.error(
                        ploc,
                        (
                            "Unknown panel status: "
                            f"{panel_status}"
                        ),
                    )

                sources = panel.get(
                    "sources",
                    [],
                )

                if not isinstance(
                    sources,
                    list,
                ):
                    self.error(
                        ploc,
                        "sources must be a list",
                    )
                    sources = []

                for source_id in sources:

                    if source_id not in datasets:
                        self.error(
                            ploc,
                            (
                                "Unknown dataset "
                                f"source: {source_id}"
                            ),
                        )

                metric_ids = panel.get(
                    "metrics",
                    [],
                )

                if not isinstance(
                    metric_ids,
                    list,
                ):
                    self.error(
                        ploc,
                        "metrics must be a list",
                    )
                    metric_ids = []

                for metric_id in metric_ids:

                    if metric_id not in metrics:
                        self.error(
                            ploc,
                            (
                                "Unknown metric: "
                                f"{metric_id}"
                            ),
                        )

                pairing = panel.get(
                    "pairing",
                    "none",
                )

                if (
                    allowed_pairings
                    and pairing
                    not in allowed_pairings
                ):
                    self.error(
                        ploc,
                        (
                            "Unknown pairing: "
                            f"{pairing}"
                        ),
                    )

                uncertainty = panel.get(
                    "uncertainty",
                    {},
                )

                if isinstance(
                    uncertainty,
                    dict,
                ):

                    method = uncertainty.get(
                        "method"
                    )

                    if (
                        method
                        and allowed_uncertainty
                        and method
                        not in allowed_uncertainty
                    ):
                        self.error(
                            ploc,
                            (
                                "Unknown uncertainty "
                                f"method: {method}"
                            ),
                        )

                elif uncertainty is not None:
                    self.error(
                        ploc,
                        (
                            "uncertainty must "
                            "be a mapping"
                        ),
                    )

                self.validate_manuscript_safety(
                    figure_key=key,
                    figure_spec=spec,
                    panel_id=str(panel_id),
                    source_ids=sources,
                )

    # ------------------------------------------------------------------
    # Manuscript safety
    # ------------------------------------------------------------------

    def validate_manuscript_safety(
        self,
        figure_key: str,
        figure_spec: dict[str, Any],
        panel_id: str,
        source_ids: list[str],
    ):

        policy = (
            self.contract
            .get(
                "global_policy",
                {},
            )
            .get(
                "manuscript",
                {},
            )
        )

        allowed_groups = set(
            policy.get(
                "allowed_groups",
                [],
            )
        )

        forbidden_kinds = set(
            policy.get(
                "forbidden_dataset_kinds",
                [],
            )
        )

        forbidden_lifecycles = set(
            policy.get(
                "forbidden_dataset_lifecycles",
                [],
            )
        )

        group = figure_spec.get(
            "group"
        )

        if group not in allowed_groups:
            return

        datasets = self.contract.get(
            "datasets",
            {},
        )

        ploc = (
            f"figures.{figure_key}."
            f"panels.{panel_id}"
        )

        for source_id in source_ids:

            source = datasets.get(
                source_id,
                {},
            )

            if not source:
                continue

            if not bool(
                source.get(
                    "manuscript_allowed",
                    False,
                )
            ):
                self.error(
                    ploc,
                    (
                        "Manuscript figure "
                        "references blocked "
                        f"dataset: {source_id}"
                    ),
                )

            kind = source.get(
                "kind"
            )

            if kind in forbidden_kinds:
                self.error(
                    ploc,
                    (
                        "Manuscript figure "
                        "references forbidden "
                        f"dataset kind {kind}: "
                        f"{source_id}"
                    ),
                )

            lifecycle = (
                self._dataset_lifecycle(
                    source
                )
            )

            if lifecycle in forbidden_lifecycles:
                self.error(
                    ploc,
                    (
                        "Manuscript figure "
                        "references forbidden "
                        "dataset lifecycle "
                        f"{lifecycle}: "
                        f"{source_id}"
                    ),
                )

    # ------------------------------------------------------------------
    # Builder validation
    # ------------------------------------------------------------------

    def validate_builder_string(
        self,
        location: str,
        builder: str,
    ):

        if ":" not in builder:
            self.error(
                location,
                (
                    "Builder must use "
                    "'module:function' format"
                ),
            )
            return

        module, function = builder.split(
            ":",
            1,
        )

        if not module or not function:
            self.error(
                location,
                (
                    "Malformed builder: "
                    f"{builder}"
                ),
            )

    def validate_builder_import(
        self,
        location: str,
        builder: str,
    ):

        if ":" not in builder:
            return

        module_name, function_name = (
            builder.split(
                ":",
                1,
            )
        )

        try:
            module = importlib.import_module(
                module_name
            )

        except Exception as exc:
            self.error(
                location,
                (
                    "Cannot import builder "
                    f"module {module_name}: "
                    f"{exc}"
                ),
            )
            return

        fn = getattr(
            module,
            function_name,
            None,
        )

        if fn is None:
            self.error(
                location,
                (
                    "Builder function missing: "
                    f"{builder}"
                ),
            )

        elif not callable(
            fn
        ):
            self.error(
                location,
                (
                    "Builder is not callable: "
                    f"{builder}"
                ),
            )

    # ------------------------------------------------------------------
    # Run all checks
    # ------------------------------------------------------------------

    def run(self):

        self.validate_top_level()
        self.validate_metrics()
        self.validate_datasets()
        self.validate_figures()

        return self.messages


# ======================================================================
# Public API
# ======================================================================

def load_contract(
    path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        contract = yaml.safe_load(
            f
        )

    if contract is None:
        raise ValueError(
            f"Empty contract: {path}"
        )

    if not isinstance(
        contract,
        dict,
    ):
        raise TypeError(
            (
                "Figure contract root "
                "must be a mapping"
            )
        )

    return contract


def validate_contract(
    path: Path = DEFAULT_CONTRACT,
    root: Path = ROOT,
    check_files: bool = True,
    check_builders: bool = False,
):

    contract = load_contract(
        path
    )

    validator = ContractValidator(
        contract=contract,
        root=root,
        check_files=check_files,
        check_builders=check_builders,
    )

    return validator.run()


# ======================================================================
# CLI
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Validate AMAF paper "
            "figure contract"
        )
    )

    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT,
    )

    parser.add_argument(
        "--skip-files",
        action="store_true",
        help=(
            "Validate schema without "
            "checking dataset files"
        ),
    )

    parser.add_argument(
        "--check-builders",
        action="store_true",
        help=(
            "Import builder modules/functions. "
            "Use after plotting modules exist."
        ),
    )

    args = parser.parse_args()

    try:
        messages = validate_contract(
            path=args.contract,
            root=ROOT,
            check_files=not args.skip_files,
            check_builders=args.check_builders,
        )

    except Exception as exc:

        print(
            "FIGURE CONTRACT: FAIL"
        )

        print(
            f"Fatal error: {exc}"
        )

        return 1

    errors = [
        x
        for x in messages
        if x.level == "ERROR"
    ]

    warnings = [
        x
        for x in messages
        if x.level == "WARNING"
    ]

    infos = [
        x
        for x in messages
        if x.level == "INFO"
    ]

    for msg in messages:

        print(
            f"[{msg.level:<7}] "
            f"{msg.location}: "
            f"{msg.message}"
        )

    print()
    print(
        "----------------------------------------"
    )

    print(
        f"errors:   {len(errors)}"
    )

    print(
        f"warnings: {len(warnings)}"
    )

    print(
        f"info:     {len(infos)}"
    )

    if errors:

        print()
        print(
            "FIGURE CONTRACT: FAIL"
        )

        return 1

    print()
    print(
        "FIGURE CONTRACT: PASS"
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
