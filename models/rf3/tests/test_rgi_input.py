"""Per-input RGI configuration and component metadata survive RF3 parsing."""

import json
from copy import deepcopy

import numpy as np
import pytest
from rf3.utils.inference import InferenceInput, prepare_inference_inputs_from_paths


def test_component_flags_follow_expanded_and_automatic_chains():
    data = {
        "name": "chains",
        "components": [
            {"sequence": "GLKE", "id": ["A", "C"], "conformer_restraints": True},
            {"smiles": "C[N+](C)(C)C", "conformer_restraints": True},
            {"seq": "GLKE", "chain_id": "Z"},
        ],
        "restraints_config": {"conformer_restraints_config": {}},
    }
    original = deepcopy(data)
    spec = InferenceInput.from_json_dict(data)
    aa = spec.to_pipeline_input()["atom_array"]
    assert set(spec.conformer_restraints) == set(aa.chain_id)
    assert spec.conformer_restraints["A"] is True
    assert spec.conformer_restraints["C"] is True
    assert spec.conformer_restraints["Z"] is False
    assert all(aa.conformer_restraints[aa.chain_id != "Z"])
    assert not any(aa.conformer_restraints[aa.chain_id == "Z"])
    assert data == original
    assert "conformer_restraints" not in spec.atom_array.get_annotation_categories()


def test_input_configs_are_independent_and_omission_is_disabled():
    base = {"name": "first", "components": [{"seq": "GLKE", "chain_id": "A"}]}
    first = InferenceInput.from_json_dict(
        {**base, "restraints_config": {"max_iter": 11}}
    )
    second = InferenceInput.from_json_dict(
        {**base, "restraints_config": {"max_iter": 29}}
    )
    omitted = InferenceInput.from_json_dict(base)
    first.restraints_config["max_iter"] = 7
    assert second.restraints_config == {"max_iter": 29}
    assert omitted.restraints_config is None


def test_json_config_path_is_relative_to_input_file(tmp_path, monkeypatch):
    config_dir = tmp_path / "inputs"
    config_dir.mkdir()
    (config_dir / "restraints.yaml").write_text("max_iter: 17\n")
    input_file = config_dir / "jobs.json"
    input_file.write_text(
        json.dumps(
            [
                {
                    "name": "relative",
                    "components": [{"seq": "GLKE"}],
                    "restraints_config": {"config_path": "restraints.yaml"},
                }
            ]
        )
    )
    monkeypatch.chdir(tmp_path)
    specs = prepare_inference_inputs_from_paths([input_file])
    assert len(specs) == 1
    assert specs[0].restraints_config == {"max_iter": 17}


@pytest.mark.parametrize("flag", ["false", 1, None, []])
def test_invalid_component_flag_is_rejected(flag):
    with pytest.raises(TypeError, match="boolean"):
        InferenceInput.from_json_dict(
            {
                "name": "invalid",
                "components": [{"seq": "GLKE", "conformer_restraints": flag}],
            }
        )


def test_python_api_and_unknown_chain():
    spec = InferenceInput.from_json_dict(
        {
            "name": "source",
            "components": [{"seq": "GLKE", "chain_id": "A"}],
        }
    )
    copy = InferenceInput.from_atom_array(
        spec.atom_array,
        restraints_config={"max_iter": 13},
        conformer_restraints={"A": True},
    )
    assert copy.restraints_config == {"max_iter": 13}
    assert np.all(copy.to_pipeline_input()["atom_array"].conformer_restraints)
    copy.conformer_restraints = {"missing": True}
    with pytest.raises(ValueError, match="Unknown conformer restraint chains"):
        copy.to_pipeline_input()
