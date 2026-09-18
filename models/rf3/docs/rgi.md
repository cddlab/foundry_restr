# Restraint-Guided Inference (RGI)

This fork integrates RF3 with [RGI-toolkit](https://github.com/cddlab/rgi_toolkit).
RGI minimizes user-defined energies on the denoised coordinates at each diffusion
step, before RF3's integrator update. The integration applies to **RF3 only**.

## Installation

Run from this fork's checkout on the `rgi-integration` branch:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend cu128 \
  -c constraints-rf3.txt -e '.[rf3]'
uv run --no-project --python .venv/bin/python foundry install rf3
```

The fork installs the shared RGI engine. `constraints-rf3.txt` pins the validated
Linux CUDA stack, including compatible PyTorch and cuEquivariance versions.
For development with sibling checkouts,
install the matching engine source into the same environment:

```bash
uv pip install --python .venv/bin/python -e ../RGI-toolkit
```

Use a CUDA-enabled PyTorch installation for GPU inference. Submit environment
installation and inference through your scheduler when required by your cluster.

## Input JSON

Each JSON job accepts one `restraints_config` dictionary. All configuration is
parsed by RGI-toolkit; its
[configuration reference](https://github.com/cddlab/rgi_toolkit/blob/main/docs/config.md)
is the source of truth for types, defaults, selection syntax and optimizer options.

```json
{
  "name": "restrained_peptide",
  "components": [{"seq": "GLKEMALQ", "chain_id": "A"}],
  "restraints_config": {
    "verbose": true,
    "distance_restraints_config": [{
      "atom_selection1": "chain A and resid 1 and name CA",
      "atom_selection2": "chain A and resid 8 and name CA",
      "harmonic": {"target_distance": 12.0}
    }]
  }
}
```

```bash
uv run --no-project --python .venv/bin/python rf3 fold \
  inputs=job.json out_dir=predictions diffusion_batch_size=2 \
  num_steps=50 early_stopping_plddt_threshold=null
```

The shared engine supports distance, angle, dihedral, improper, chiral, plane,
RMSD, Watson–Crick base-pair, custom and reference-anchored restraints, together
with ligand and polymer conformer terms. There are no separate restraint CLI flags.
Omit `restraints_config` or set it to `null` to retain RF3's ordinary sampling path.
An empty dictionary adds no restraints.

External JSON/YAML configuration is also supported:

```json
"restraints_config": {"config_path": "restraints.yaml"}
```

`config_path` is relative to the containing input JSON file. Paths inside an
external configuration are relative to that configuration file. Inline reference
paths retain the shared engine's working-directory semantics.

## Selections and timing

- `chain` uses the chain ID present in the processed structure. Set explicit
  component `chain_id` values to make selections predictable.
- `resid` is the one-based token ordinal **within each chain**, not the author
  residue number. Standard polymer residues occupy one token; atomized ligands
  and modified residues may occupy one token per atom.
- `index` is the zero-based row in RF3's final, processed atom array, which is the
  same order as its diffusion coordinates. Avoid input-file atom indices after
  filtering or tokenization.
- Sigma windows use the schedule sigma **before churn**, matching other RGI
  integrations. Step windows use the zero-based index of the actual rollout;
  a partial-diffusion rollout starts at step zero. Sigma and step windows are
  mutually exclusive on one restraint.

## Conformer restraints

Both the shared conformer configuration and a per-component opt-in are required:

```json
{
  "name": "ligand_geometry",
  "components": [
    {"seq": "GLKEMALQ", "chain_id": "A"},
    {"ccd_code": "ATP", "chain_id": "B", "conformer_restraints": true}
  ],
  "restraints_config": {
    "verbose": true,
    "conformer_restraints_config": {
      "start_sigma": 2.0,
      "plane": {"weight": 1.0},
      "vdw": {"mode": "both", "weight": 1.0}
    }
  }
}
```

The flag also supports polymer components and replicated `chain_id` lists. It is
independent of RF3's native `ground_truth_conformer_selection`. RGI uses the
reference-conformer features rather than the coordinates being diffused.
Use `monomer_library` in the shared conformer configuration when dictionary
geometry is needed for polymers; it is never enabled implicitly.

## Python and structure-file inputs

Pass the settings on the per-structure `InferenceInput`:

```python
from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.utils.inference import InferenceInput

query = InferenceInput.from_cif_path(
    "complex.cif",
    restraints_config={"conformer_restraints_config": {}},
    conformer_restraints={"B": True},
)
engine = RF3InferenceEngine(early_stopping_plddt_threshold=None)
outputs = engine.run(inputs=query, out_dir="predictions")
```

`InferenceInput.from_atom_array()` accepts the same arguments. Python dictionary
configuration paths are relative to the working directory. JSON file components
can also carry `conformer_restraints: true` to opt in their chains.

Each structure receives its own `CombinedRestraints` instance. All diffusion
samples for that structure share its specification, and each sample is optimized
independently. A following structure cannot inherit the previous one's settings.
RF3's existing `skip_existing` behavior remains available; use
`skip_existing=false` when comparing configurations in the same output directory.

## Verification

See [the validation report](rgi-validation.md) for measured real-checkpoint
results, the tested dependency stack and repeatable CPU/GPU commands.

The runnable examples are in `examples/rgi/`:

```bash
bash examples/rgi/run.sh distance.json distance
bash examples/rgi/run.sh conformer.json conformer
```

The distance example contains two jobs with different targets (12 and 20
Angstrom), exercising configuration isolation. Inspect each written CIF and
every sample, not only the top-ranked model. With `verbose: true`, check nonzero
`built spec` counts for every requested term and inspect the `finalize` residuals.
A zero energy for an absent term does not demonstrate that a restraint ran.
The final RF3 integrator update follows the last minimization, so final coordinate
residuals need not be exactly zero. The examples test restraint behavior and do
not establish prediction accuracy or chemical validity for an arbitrary target.
