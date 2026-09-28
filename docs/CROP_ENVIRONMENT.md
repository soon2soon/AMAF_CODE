# Crop / gym-DSSAT environment

The crop runner targets the external `gym_dssat_pdi:GymDssatPdi-v0`
environment. DSSAT/PDI binaries, crop model data, and the Python environment
used for the reported runs are not redistributed in this repository.

The frozen configuration expects the DSSAT launcher at:

```text
/opt/dssat_pdi/run_dssat
```

This is a system installation path, not a developer-specific path. Before
running the crop matrix:

1. Install and validate DSSAT/PDI and `gym_dssat_pdi` independently.
2. Confirm that one environment reset and step succeeds.
3. Install AMAF without forcing dependency upgrades in that known-good
   environment.
4. Run a two-episode smoke test before the five-seed matrix.

```bash
python -m pip install -e . --no-deps
amaf-run \
  --experiment experiments/revision/crop_multiseed.yaml \
  --algorithm dqn \
  --seed 11 \
  --set training.max_episodes=2
```

The historical crop host used a legacy Python/Gym stack, whereas the public
package and analysis tools target Python 3.10+. Treat crop execution as an
external-environment integration and document any compatibility changes. The
included `docker/gym/Dockerfile` is for Gymnasium/MuJoCo and does not provide
DSSAT.
