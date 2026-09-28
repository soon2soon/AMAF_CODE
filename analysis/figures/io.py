#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from .validation import (
    DEFAULT_CONTRACT,
    ROOT,
    load_contract,
)


# ======================================================================
# Exceptions
# ======================================================================

class FigureDataError(RuntimeError):
    """Base exception for figure-data access errors."""


class UnknownDatasetError(FigureDataError):
    """Raised when a dataset ID is not registered in the contract."""


class ManuscriptSafetyError(FigureDataError):
    """Raised when a blocked/live/planned source is used for manuscript."""


# ======================================================================
# Contract registry
# ======================================================================

class FigureDataRegistry:
    """
    Contract-aware data access layer for manuscript figures.

    Figure builders should request datasets by contract ID rather than
    directly hard-coding filesystem paths.
    """

    def __init__(
        self,
        contract_path: Path = DEFAULT_CONTRACT,
        root: Path = ROOT,
    ):
        self.contract_path = Path(contract_path)
        self.root = Path(root)

        self.contract = load_contract(
            self.contract_path
        )

        self.datasets = self.contract.get(
            "datasets",
            {},
        )

        self.figures = self.contract.get(
            "figures",
            {},
        )

        self.metrics = self.contract.get(
            "metrics",
            {},
        )

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    def dataset_spec(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:

        if dataset_id not in self.datasets:
            raise UnknownDatasetError(
                f"Unknown dataset ID: {dataset_id}"
            )

        return self.datasets[
            dataset_id
        ]

    def dataset_kind(
        self,
        dataset_id: str,
    ) -> str:

        return str(
            self.dataset_spec(
                dataset_id
            ).get(
                "kind",
                ""
            )
        )

    def dataset_lifecycle(
        self,
        dataset_id: str,
    ) -> str:

        spec = self.dataset_spec(
            dataset_id
        )

        explicit = spec.get(
            "lifecycle"
        )

        if explicit:
            return str(
                explicit
            )

        defaults = {
            "canonical_csv": "canonical",
            "analysis_csv": "canonical",
            "frozen_dir": "frozen",
            "live_output": "live",
        }

        return defaults.get(
            str(
                spec.get(
                    "kind",
                    ""
                )
            ),
            "canonical",
        )

    def dataset_path(
        self,
        dataset_id: str,
    ) -> Path:

        spec = self.dataset_spec(
            dataset_id
        )

        raw = spec.get(
            "path"
        )

        if not raw:
            raise FigureDataError(
                f"Dataset has no path: {dataset_id}"
            )

        return (
            self.root
            / str(raw)
        )

    # ------------------------------------------------------------------
    # Manuscript safety
    # ------------------------------------------------------------------

    def assert_manuscript_allowed(
        self,
        dataset_id: str,
    ) -> None:

        spec = self.dataset_spec(
            dataset_id
        )

        allowed = bool(
            spec.get(
                "manuscript_allowed",
                False,
            )
        )

        lifecycle = (
            self.dataset_lifecycle(
                dataset_id
            )
        )

        kind = self.dataset_kind(
            dataset_id
        )

        if not allowed:
            raise ManuscriptSafetyError(
                (
                    f"Dataset '{dataset_id}' is "
                    "blocked from manuscript figures."
                )
            )

        forbidden_lifecycles = {
            "planned",
            "live",
            "retired",
        }

        if lifecycle in forbidden_lifecycles:
            raise ManuscriptSafetyError(
                (
                    f"Dataset '{dataset_id}' has "
                    f"forbidden manuscript lifecycle "
                    f"'{lifecycle}'."
                )
            )

        if kind == "live_output":
            raise ManuscriptSafetyError(
                (
                    f"Dataset '{dataset_id}' is "
                    "a live output and cannot be used "
                    "for manuscript figures."
                )
            )

    # ------------------------------------------------------------------
    # Path access
    # ------------------------------------------------------------------

    def load_path(
        self,
        dataset_id: str,
        *,
        manuscript: bool = False,
        must_exist: bool = True,
    ) -> Path:

        if manuscript:
            self.assert_manuscript_allowed(
                dataset_id
            )

        path = self.dataset_path(
            dataset_id
        )

        if (
            must_exist
            and not path.exists()
        ):
            raise FileNotFoundError(
                (
                    f"Dataset '{dataset_id}' "
                    f"not found: {path}"
                )
            )

        return path

    # ------------------------------------------------------------------
    # CSV access
    # ------------------------------------------------------------------

    def load_csv(
        self,
        dataset_id: str,
        *,
        manuscript: bool = False,
        validate_columns: bool = True,
        **read_csv_kwargs,
    ) -> pd.DataFrame:

        kind = self.dataset_kind(
            dataset_id
        )

        if kind not in {
            "canonical_csv",
            "analysis_csv",
        }:
            raise FigureDataError(
                (
                    f"Dataset '{dataset_id}' "
                    f"is kind '{kind}', not CSV."
                )
            )

        path = self.load_path(
            dataset_id,
            manuscript=manuscript,
            must_exist=True,
        )

        df = pd.read_csv(
            path,
            **read_csv_kwargs,
        )

        if validate_columns:
            self._assert_required_columns(
                dataset_id,
                df,
            )

        return df

    def _assert_required_columns(
        self,
        dataset_id: str,
        df: pd.DataFrame,
    ) -> None:

        spec = self.dataset_spec(
            dataset_id
        )

        required = spec.get(
            "required_columns",
            [],
        )

        missing = [
            c
            for c in required
            if c not in df.columns
        ]

        if missing:
            raise FigureDataError(
                (
                    f"Dataset '{dataset_id}' "
                    "is missing required columns: "
                    + ", ".join(missing)
                )
            )

    # ------------------------------------------------------------------
    # Generic loader
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        dataset_id: str,
        *,
        manuscript: bool = False,
    ):

        kind = self.dataset_kind(
            dataset_id
        )

        if kind in {
            "canonical_csv",
            "analysis_csv",
        }:
            return self.load_csv(
                dataset_id,
                manuscript=manuscript,
            )

        return self.load_path(
            dataset_id,
            manuscript=manuscript,
        )

    # ------------------------------------------------------------------
    # Metric / figure contract access
    # ------------------------------------------------------------------

    def metric_spec(
        self,
        metric_id: str,
    ) -> dict[str, Any]:

        if metric_id not in self.metrics:
            raise KeyError(
                f"Unknown metric ID: {metric_id}"
            )

        return self.metrics[
            metric_id
        ]

    def figure_spec(
        self,
        figure_key: str,
    ) -> dict[str, Any]:

        if figure_key not in self.figures:
            raise KeyError(
                f"Unknown figure key: {figure_key}"
            )

        return self.figures[
            figure_key
        ]

    # ------------------------------------------------------------------
    # Registry inspection
    # ------------------------------------------------------------------

    def list_datasets(
        self,
    ) -> list[str]:

        return sorted(
            self.datasets
        )

    def list_figures(
        self,
    ) -> list[str]:

        return sorted(
            self.figures
        )

    # ------------------------------------------------------------------
    # Source fingerprints
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256_file(
        path: Path,
    ) -> str:

        h = hashlib.sha256()

        with path.open(
            "rb"
        ) as f:

            while True:

                block = f.read(
                    1024 * 1024
                )

                if not block:
                    break

                h.update(
                    block
                )

        return h.hexdigest()

    def fingerprint(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:
        """
        Produce a lightweight source fingerprint for figure provenance.

        For CSV files:
            hash the CSV itself.

        For frozen directories:
            prefer SHA256SUMS, then MANIFEST.json.

        For live/planned directories:
            report lifecycle/path without pretending they are frozen.
        """

        spec = self.dataset_spec(
            dataset_id
        )

        path = self.dataset_path(
            dataset_id
        )

        kind = self.dataset_kind(
            dataset_id
        )

        lifecycle = (
            self.dataset_lifecycle(
                dataset_id
            )
        )

        out = {
            "dataset_id": dataset_id,
            "path": str(
                path.relative_to(
                    self.root
                )
            )
            if path.is_absolute()
            and self.root in path.parents
            else str(path),
            "kind": kind,
            "lifecycle": lifecycle,
            "exists": path.exists(),
        }

        if not path.exists():
            out["fingerprint_type"] = None
            out["sha256"] = None
            return out

        if path.is_file():

            out["fingerprint_type"] = (
                "file_sha256"
            )

            out["sha256"] = (
                self._sha256_file(
                    path
                )
            )

            return out

        if path.is_dir():

            sha_manifest = (
                path
                / "SHA256SUMS"
            )

            manifest_json = (
                path
                / "MANIFEST.json"
            )

            if sha_manifest.exists():

                out["fingerprint_type"] = (
                    "SHA256SUMS_sha256"
                )

                out["fingerprint_source"] = (
                    str(
                        sha_manifest.relative_to(
                            self.root
                        )
                    )
                )

                out["sha256"] = (
                    self._sha256_file(
                        sha_manifest
                    )
                )

                return out

            if manifest_json.exists():

                out["fingerprint_type"] = (
                    "MANIFEST_json_sha256"
                )

                out["fingerprint_source"] = (
                    str(
                        manifest_json.relative_to(
                            self.root
                        )
                    )
                )

                out["sha256"] = (
                    self._sha256_file(
                        manifest_json
                    )
                )

                return out

        out["fingerprint_type"] = None
        out["sha256"] = None

        return out


# ======================================================================
# Convenience singleton-style API
# ======================================================================

def registry(
    contract_path: Path = DEFAULT_CONTRACT,
) -> FigureDataRegistry:

    return FigureDataRegistry(
        contract_path=contract_path,
        root=ROOT,
    )


def load_csv(
    dataset_id: str,
    *,
    manuscript: bool = False,
) -> pd.DataFrame:

    return registry().load_csv(
        dataset_id,
        manuscript=manuscript,
    )


def load_path(
    dataset_id: str,
    *,
    manuscript: bool = False,
) -> Path:

    return registry().load_path(
        dataset_id,
        manuscript=manuscript,
    )


def load_dataset(
    dataset_id: str,
    *,
    manuscript: bool = False,
):

    return registry().load_dataset(
        dataset_id,
        manuscript=manuscript,
    )


def source_fingerprint(
    dataset_id: str,
) -> dict[str, Any]:

    return registry().fingerprint(
        dataset_id
    )
