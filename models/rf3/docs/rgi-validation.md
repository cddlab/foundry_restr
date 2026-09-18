# RF3 RGI validation

These checks use the real `rf3_foundry_01_24_latest_remapped.ckpt` checkpoint,
the native `rf3 fold` CLI, and the shared RGI optimizer. They verify the RF3
integration and restraint behavior, not general prediction accuracy.

## Environment and protocol

- Linux, Python 3.12.3, NVIDIA RTX 4090 (24 GB).
- PyTorch 2.7.1+cu128, cuEquivariance 0.6.1, AtomWorks 2.2.1,
  RDKit 2025.3.6, Biotite 1.4.0, NumPy 2.5.3.
- `diffusion_batch_size=2`, `n_recycles=2`, `num_steps=50`, `seed=42`,
  `early_stopping_plddt_threshold=null`, `skip_existing=false`.
- Installation and validation ran through Slurm, with at most two concurrent
  nodes: one GPU allocation and one CPU allocation.

Use [the installation guide](rgi.md) and `constraints-rf3.txt`. Unconstrained
cuEquivariance 0.11 is incompatible with the pinned PyTorch 2.7.1 stack; 0.6.1
is the combination exercised here.

## Real-model results

All six E2E checks passed. Eleven input structures produced 22 individually
checked, finite-coordinate sample CIFs. Measurements below are recomputed from
those CIFs, independently of the optimizer's energy logger.

| Check | Result across both samples |
| --- | --- |
| Two inputs in one run, CA distance targets 12 and 20 Å | Absolute error below 0.00003 Å for each target; configurations stayed independent |
| Standard QBP domain-centroid target 25 Å | 24.99909 and 25.00058 Å |
| DNA G–C Watson–Crick pair | All three hydrogen-bond distances within the configured flat-bottom window; observed range 2.72249–2.83998 Å |
| Omitted versus empty configuration, same seed | Corresponding atom coordinates agree within 0.002 Å |
| ATP and fumarate conformer restraints | ATP source stereocenters retained; fumarate E geometry retained in both samples |
| Standalone angle, plane, custom distance and reference RMSD | Angle error below 0.001°; plane RMS deviation below 0.00002 Å; custom distance error below 0.00002 Å; aligned RMSD below 0.00003 Å |

The conformer specification contained 40 bonds, 60 angles, 12 chirals, four
planes and one cis/trans term. VdW included 390 intraligand pairs, 248
interligand pairs, and a 39-ligand-atom / 60-background-atom neighbor setup.
Final logged bond and angle energies were 0.00004 and 0.00007; chiral, plane,
cis/trans and VdW energies rounded to zero. Torsion was not requested and its
zero count is not evidence of validation.

The DNA fixture uses four residues per chain because RF3's existing preprocessing
removes shorter polymer chains. JSON chain types use AtomWorks' full
`POLYDEOXYRIBONUCLEOTIDE` name.

## CPU and documentation checks

- Foundry CPU regression suite: **741 passed**, 31 GPU/integration tests deselected.
- Shared RF3 adapter and existing adapter chemistry tests: **79 passed**.
- Toolkit NumPy/PyTorch/JAX backend parity: **25 passed**.
- Foundry mypy: **239 source files passed**.
- Changed Python files passed each repository's Ruff checks and formatting.
- Sphinx HTML build succeeded. Seven existing RFD3 documentation warnings remain;
  the new RF3 pages introduced none.

## Reproduce

Run inside the appropriate scheduler allocation after installing the dependencies
and checkpoint. Set `RF3_CKPT_PATH` to the checkpoint file if it is outside the
normal Foundry cache.

```bash
# CPU regression and type checking, from the Foundry checkout.
uv run --no-project --python .venv/bin/python python -m pytest \
  -m 'not gpu and not integration' -q
uv run --no-project --python .venv/bin/python python -m mypy

# Real-checkpoint E2E tests, from a GPU allocation.
uv run --no-project --python .venv/bin/python python -m pytest \
  models/rf3/tests/integration/test_rgi.py -v \
  --basetemp=.rgi-validation/e2e

# Documentation.
uv run --no-project --python .venv/bin/python python -m sphinx \
  -b html docs/source .rgi-validation/docs
```

The E2E tests retain each CLI log, input JSON, predicted CIF and
`measurements.json` beneath the selected `--basetemp`. That directory is
reproducible output and is ignored by Git. Pytest replaces a reused `--basetemp`;
choose a different directory to preserve an earlier run.

From the sibling toolkit checkout, using its development environment:

```bash
uv run --no-project --python .venv/bin/python python -m pytest \
  tests/test_rf3_adapter.py tests/test_adapters_shared.py tests/test_backend_parity.py -q
```
