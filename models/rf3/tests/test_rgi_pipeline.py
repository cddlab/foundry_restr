"""Real AtomWorks preprocessing must preserve the RGI coordinate correspondence."""

import json
from pathlib import Path

import numpy as np
import pytest
from rf3.data.pipelines import build_af3_transform_pipeline
from rf3.utils.inference import InferenceInput
from rf3.utils.restraints import build_rf3_restraints


@pytest.fixture(scope="module")
def pipeline():
    return build_af3_transform_pipeline(
        is_inference=True,
        protein_msa_dirs=[],
        rna_msa_dirs=[],
        n_recycles=1,
        diffusion_batch_size=2,
        run_confidence_head=True,
        residue_cache_dir=None,
        undesired_res_names=[],
        use_element_for_atom_names_of_atomized_tokens=True,
    )


def test_ligand_chemistry_survives_real_preprocessing(pipeline):
    root = Path(__file__).resolve().parents[3]
    data = json.loads((root / "examples/rgi/conformer.json").read_text())
    query = InferenceInput.from_json_dict(data)
    batch = pipeline(query.to_pipeline_input())
    restraints = build_rf3_restraints(query, batch)
    spec = restraints.spec
    for field in ["bond", "angle", "chiral", "plane", "cistrans", "vdw"]:
        assert int(getattr(spec, field).mask.sum()) > 0, field
    assert spec.vdw_config is not None
    assert len(spec.vdw_config.background_global) > 0
    assert np.all(spec.active_sites < len(batch["atom_array"]))
    assert restraints.is_active()


def test_distance_setup_is_independent_of_following_inputs(pipeline):
    def query(target):
        return InferenceInput.from_json_dict(
            {
                "name": f"distance_{target}",
                "components": [{"seq": "GLKEMALQ", "chain_id": "A"}],
                "restraints_config": {
                    "gpu": False,
                    "distance_restraints_config": [
                        {
                            "atom_selection1": "chain A and resid 1 and name CA",
                            "atom_selection2": "chain A and resid 8 and name CA",
                            "harmonic": {"target_distance": target},
                        }
                    ],
                },
            }
        )

    first, second = query(12), query(20)
    batch = pipeline(first.to_pipeline_input())
    a = build_rf3_restraints(first, batch)
    b = build_rf3_restraints(second, batch)
    assert a is not b
    coords = batch["feats"]["ref_pos"].unsqueeze(0).repeat(2, 1, 1).clone()
    aa = batch["atom_array"]
    ca = np.flatnonzero((aa.chain_id == "A") & (aa.atom_name == "CA"))
    for instance, target in [(a, 12), (b, 20), (a, 12)]:
        result = instance.minimize(coords.clone(), 0, 0.5).numpy()
        distance = np.linalg.norm(result[:, ca[0]] - result[:, ca[-1]], axis=-1)
        np.testing.assert_allclose(distance, target, atol=1e-3)
    first.restraints_config = None
    assert build_rf3_restraints(first, batch) is None
