# Public release scope

This repository is a clean code-and-derived-artifact release. It was assembled
without importing the private development repository's Git history.

## Included

- Importable AMAF training and analysis package.
- Final revision experiment matrices and their transitive YAML configuration
  dependencies.
- Unit tests and a Gym/MuJoCo container definition.
- Paper-analysis source code, canonical derived tables, Results/Discussion
  handoff tables, plot-ready figure data, and Gnuplot sources.
- Data checksums, citation metadata, dependency declarations, and runnable
  examples.

## Deliberately excluded

- Raw per-step/per-episode frozen trajectories (approximately 127 MB in the
  development workspace).
- Model and optimizer checkpoints.
- Legacy notebooks and superseded protocol configurations.
- Machine-specific environment captures, generated DSSAT files, host paths,
  and internal provenance/audit reports.
- Generated PDF/PNG figures and compressed duplicate archives.
- Private branch names, remotes, and development Git history.

The included `paper_tables/` and plot-ready data are sufficient to inspect the
reported aggregates and render the manuscript figures. They are not a
replacement for the omitted raw trajectory archive. The raw-to-canonical
analysis source is retained so that the full audit can be repeated when the
frozen raw archive is supplied separately.

## Integrity boundary

`DATA_SHA256SUMS` covers every released CSV and TSV data file. Verify it from
the repository root with:

```bash
sha256sum -c DATA_SHA256SUMS
```

The checksum manifest protects the public derived-data snapshot; it does not
claim that omitted checkpoints or raw trajectories are present.
