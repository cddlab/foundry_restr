"""Real-checkpoint RF3 CLI checks for RGI, including every written sample.

Run with RF3_CKPT_PATH set and a GPU allocation. These tests use real model
weights and shared RGI optimization; no network or sampler is mocked.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module", autouse=True)
def require_model_and_gpu(gpu, require_ckpt):
    """Keep real-model checks out of portable CPU unit-test runs."""


def fold(jobs, output, *, steps=50):
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "inputs.json"
    input_path.write_text(json.dumps(jobs, indent=2))
    checkpoint = os.environ.get("RF3_CKPT_PATH", "rf3")
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("rf3")),
            "fold",
            f"inputs={input_path}",
            f"out_dir={output / 'predictions'}",
            f"ckpt_path={checkpoint}",
            "diffusion_batch_size=2",
            "n_recycles=2",
            f"num_steps={steps}",
            "seed=42",
            "skip_existing=false",
            "early_stopping_plddt_threshold=null",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    log = result.stdout + "\n" + result.stderr
    (output / "run.log").write_text(log)
    assert result.returncode == 0, log[-16000:]
    return output / "predictions", log


def samples(root, name):
    paths = sorted((root / name).glob("seed-*_sample-*/*_model.cif"))
    assert len(paths) == 2, (name, paths)
    result = []
    for path in paths:
        structure = gemmi.read_structure(str(path))
        atoms = {
            (chain.name, residue.seqid.num, atom.name): np.array(
                [atom.pos.x, atom.pos.y, atom.pos.z]
            )
            for chain in structure[0]
            for residue in chain
            for atom in residue
        }
        assert atoms and np.isfinite(list(atoms.values())).all(), path
        result.append(atoms)
    return result


def peptide(name, config=None):
    job = {"name": name, "components": [{"seq": "GLKEMALQ", "chain_id": "A"}]}
    if config is not None:
        job["restraints_config"] = {"verbose": True, **config}
    return job


def ca(residue):
    return f"chain A and resid {residue} and name CA"


@pytest.fixture(scope="module")
def distance_run(tmp_path_factory):
    jobs = json.loads((ROOT / "examples/rgi/distance.json").read_text())
    return fold(jobs, tmp_path_factory.mktemp("rgi_distance"))


def test_distance_targets_and_batch_isolation(distance_run):
    output, log = distance_run
    specs = [
        line
        for line in log.splitlines()
        if line.startswith("[rgi_toolkit] INFO built spec:")
    ]
    assert len(specs) == 2
    assert all(re.search(r"\bdistances=1(?:\s|$)", line) for line in specs)
    measurements = {}
    for name, target in [("distance_12", 12), ("distance_20", 20)]:
        distances = [
            float(np.linalg.norm(atoms[("A", 1, "CA")] - atoms[("A", 8, "CA")]))
            for atoms in samples(output, name)
        ]
        np.testing.assert_allclose(distances, target, atol=0.06)
        measurements[name] = distances
    (output.parent / "measurements.json").write_text(json.dumps(measurements, indent=2))


def test_standard_qbp_centroid_distance(tmp_path):
    job = json.loads((ROOT / "examples/rgi/qbp.json").read_text())
    output, log = fold([job], tmp_path / "qbp")
    assert re.search(r"built spec:.*distances=1(?:\s|$)", log), log[-8000:]
    distances = []
    for atoms in samples(output, "qbp_25"):
        group1 = [
            xyz
            for (chain, residue, _), xyz in atoms.items()
            if chain == "A" and (5 <= residue <= 84 or 186 <= residue <= 224)
        ]
        group2 = [
            xyz
            for (chain, residue, _), xyz in atoms.items()
            if chain == "A" and 90 <= residue <= 180
        ]
        assert len(group1) > 900 and len(group2) > 650
        distances.append(
            float(np.linalg.norm(np.mean(group1, axis=0) - np.mean(group2, axis=0)))
        )
    np.testing.assert_allclose(distances, 25, atol=0.08)
    (output.parent / "measurements.json").write_text(
        json.dumps({"qbp_centroid_distance": distances}, indent=2)
    )


def test_watson_crick_base_pair(tmp_path):
    job = {
        "name": "base_pair",
        "components": [
            {"seq": "GGAT", "chain_type": "POLYDEOXYRIBONUCLEOTIDE", "chain_id": "A"},
            {"seq": "ATCC", "chain_type": "POLYDEOXYRIBONUCLEOTIDE", "chain_id": "B"},
        ],
        "restraints_config": {
            "verbose": True,
            "base_pair_restraints_config": [
                {
                    "residue1": "chain A and resid 1",
                    "residue2": "chain B and resid 4",
                }
            ],
        },
    }
    output, log = fold([job], tmp_path / "base_pair")
    assert "base_pair=1 entries" in log
    distances = []
    for atoms in samples(output, "base_pair"):
        measured = [
            float(np.linalg.norm(atoms[("A", 1, a)] - atoms[("B", 4, b)]))
            for a, b in [("O6", "N4"), ("N1", "N3"), ("N2", "O2")]
        ]
        assert all(2.64 < value < 3.16 for value in measured), measured
        distances.append(measured)
    (output.parent / "measurements.json").write_text(
        json.dumps({"wc_distances": distances}, indent=2)
    )


@pytest.fixture(scope="module")
def native_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("rgi_native")
    native, native_log = fold([peptide("native")], root / "native")
    empty, empty_log = fold([peptide("empty", {})], root / "empty")
    return root, native, native_log, empty, empty_log


def test_empty_configuration_matches_unrestrained_model(native_runs):
    _, native, native_log, empty, empty_log = native_runs
    assert "built spec:" not in native_log
    assert "NO ACTIVE RESTRAINTS" in empty_log
    for ordinary, configured in zip(
        samples(native, "native"), samples(empty, "empty"), strict=True
    ):
        assert ordinary.keys() == configured.keys()
        np.testing.assert_allclose(
            list(ordinary.values()), list(configured.values()), atol=0.002, rtol=0
        )


def test_ligand_conformer_counts_and_stereochemistry(tmp_path):
    from atomworks.io.tools.rdkit import ccd_code_to_rdkit
    from rdkit import Chem

    job = json.loads((ROOT / "examples/rgi/conformer.json").read_text())
    output, log = fold([job], tmp_path / "conformer")
    built = next(line for line in log.splitlines() if "built spec:" in line)
    for term in ["bonds", "angles", "chirals", "plane", "cistrans"]:
        match = re.search(rf"\b{term}=(\d+)", built)
        assert match and int(match[1]) > 0, built
    assert re.search(
        r"vdw=[1-9]\d*intra\+[1-9]\d*inter\+[1-9]\d*lig/[1-9]\d*bg/", built
    ), built
    final = next(
        line for line in log.splitlines() if "finalize" in line and "bond=" in line
    )
    energies = {
        key: float(value) for key, value in re.findall(r"(\w+)=([-+\deE.]+)", final)
    }
    assert energies["bond"] < 0.5, energies
    assert energies["angle"] < 0.5, energies
    assert energies["chiral"] < 0.2, energies
    assert energies["cistrans"] < 0.02, energies
    atp = Chem.RemoveAllHs(ccd_code_to_rdkit("ATP"))
    expected_atp = dict(Chem.FindMolChiralCenters(atp, includeUnassigned=False))
    assert len(expected_atp) >= 4
    names = [atom.GetProp("atom_name") for atom in atp.GetAtoms()]
    for atoms in samples(output, job["name"]):
        predicted = Chem.Mol(atp)
        for i, name in enumerate(names):
            predicted.GetConformer().SetAtomPosition(
                i,
                next(
                    tuple(coord)
                    for (chain, _, atom_name), coord in atoms.items()
                    if chain == "B" and atom_name == name
                ),
            )
        Chem.AssignStereochemistryFrom3D(predicted, replaceExistingTags=True)
        actual = dict(
            Chem.FindMolChiralCenters(predicted, includeUnassigned=False, force=True)
        )
        assert actual == expected_atp
        # Fumarate's carboxyl carbon substituents must lie on opposite sides of C=C.
        xyz = {name: coord for (chain, _, name), coord in atoms.items() if chain == "C"}
        axis = xyz["C2"] - xyz["C1"]
        normal1 = np.cross(xyz["C0"] - xyz["C1"], axis)
        normal2 = np.cross(xyz["C3"] - xyz["C2"], axis)
        cosine = np.dot(normal1, normal2) / (
            np.linalg.norm(normal1) * np.linalg.norm(normal2)
        )
        assert cosine < -0.99, cosine
    (output.parent / "measurements.json").write_text(
        json.dumps({"energies": energies, "atp_stereo": expected_atp}, indent=2)
    )


def test_geometry_custom_and_rmsd(native_runs, tmp_path):
    _, native, _, _, _ = native_runs
    reference = native / "native/native_model.cif"
    jobs = [
        peptide(
            "angle",
            {
                "angle_restraints_config": [
                    {
                        "atom_selection1": ca(1),
                        "atom_selection2": ca(4),
                        "atom_selection3": ca(8),
                        "harmonic": {"target_angle": 90},
                        "move": "all",
                    }
                ]
            },
        ),
        peptide(
            "plane",
            {
                "plane_restraints_config": [
                    {
                        "atom_selection1": "chain A and name CA",
                        "harmonic": {"target_plane": 0},
                    }
                ]
            },
        ),
        peptide(
            "custom",
            {
                "custom_restraints_config": [
                    {
                        "energy": "(distance(A,B)-18)**2",
                        "selections": {"A": ca(1), "B": ca(8)},
                    }
                ]
            },
        ),
        peptide(
            "rmsd",
            {
                "rmsd_restraints_config": [
                    {
                        "ref_cif": str(reference),
                        "pairing": "identity",
                        "atom_selection_target": "chain A and name CA",
                        "atom_selection_ref": "chain A and name CA",
                        "harmonic": {"target_rmsd": 0},
                    }
                ]
            },
        ),
    ]
    output, log = fold(jobs, tmp_path / "geometry")
    for term in ["group_angle", "group_plane", "rmsd"]:
        assert re.search(rf"built spec:.*\b{term}=1(?:\s|$)", log), log[-16000:]
    measurements = {}
    for name in ["angle", "plane", "custom", "rmsd"]:
        measured = []
        for atoms in samples(output, name):
            coords = np.stack([atoms[("A", i, "CA")] for i in range(1, 9)])
            if name == "angle":
                v, w = coords[0] - coords[3], coords[7] - coords[3]
                value = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w)),
                                -1,
                                1,
                            )
                        )
                    )
                )
                assert abs(value - 90) < 1, value
            elif name == "plane":
                value = float(
                    np.linalg.svd(coords - coords.mean(0), compute_uv=False)[-1]
                    / np.sqrt(len(coords))
                )
                assert value < 0.03, value
            elif name == "custom":
                value = float(np.linalg.norm(coords[0] - coords[-1]))
                assert abs(value - 18) < 0.06, value
            else:
                ref_structure = gemmi.read_structure(str(reference))
                ref = np.array(
                    [
                        [a.pos.x, a.pos.y, a.pos.z]
                        for c in ref_structure[0]
                        for r in c
                        for a in r
                        if a.name == "CA"
                    ]
                )
                moving, fixed = coords - coords.mean(0), ref - ref.mean(0)
                u, _, vt = np.linalg.svd(moving.T @ fixed)
                rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
                value = float(
                    np.sqrt(np.mean(np.sum((moving @ rotation - fixed) ** 2, axis=-1)))
                )
                assert value < 0.06, value
            measured.append(value)
        measurements[name] = measured
    (output.parent / "measurements.json").write_text(json.dumps(measurements, indent=2))
