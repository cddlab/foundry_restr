"""RF3-specific input extraction for the shared RGI adapter and lifecycle."""

import numpy as np
from atomworks.enums import ChainType


def _ligand_sources(input_spec, ligand_chains):
    """Resolve original SMILES, CCD and SDF chemistry using AtomWorks readers."""
    from atomworks.io.tools.rdkit import (
        atom_array_from_rdkit,
        atom_array_to_rdkit,
        ccd_code_to_rdkit,
        sdf_to_rdkit,
    )
    from rdkit import Chem

    sources = {}
    for component in input_spec.rgi_components:
        chain = getattr(component, "chain_id", None)
        if chain not in ligand_chains:
            continue
        if hasattr(component, "smiles"):
            mol = Chem.MolFromSmiles(component.smiles)
        elif hasattr(component, "ccd_code"):
            mol = ccd_code_to_rdkit(component.ccd_code)
        elif hasattr(component, "path") and str(component.path).endswith(".sdf"):
            mol = sdf_to_rdkit(component.path)
        else:
            continue
        if mol is None:
            raise ValueError(f"Cannot read source ligand chemistry for chain {chain}")
        # Use the same naming implementation that created the input AtomArray.
        source_array = atom_array_from_rdkit(mol)
        mol = Chem.RemoveAllHs(mol)
        for atom, name in zip(mol.GetAtoms(), source_array.atom_name, strict=True):
            atom.SetProp("atom_name", str(name))
        sources[str(chain)] = mol

    # Python/CIF inputs retain chemistry on their input AtomArray, before RF3
    # replaces or randomizes any reference coordinates.
    for chain in sorted(ligand_chains - sources.keys()):
        source_array = input_spec.atom_array[input_spec.atom_array.chain_id == chain]
        sources[chain] = atom_array_to_rdkit(
            source_array, hydrogen_policy="infer", set_coord=True
        )
    return sources


def build_rf3_restraints(input_spec, pipeline_output):
    """Build one independent RGI instance from the final, CPU-side RF3 features."""
    if input_spec.restraints_config is None:
        return None

    from rgi_toolkit.combined import CombinedRestraints
    from rgi_toolkit.config import resolve_restraints_config
    from rgi_toolkit.rf3.adapter import RF3Adapter

    config = resolve_restraints_config(input_spec.restraints_config)
    aa = pipeline_output["atom_array"]
    feats = pipeline_output["feats"]
    mol_types = np.full(len(aa), None, dtype=object)
    mol_types[np.isin(aa.chain_type, ChainType.get_proteins())] = "protein"
    mol_types[aa.chain_type == ChainType.DNA] = "dna"
    mol_types[aa.chain_type == ChainType.RNA] = "rna"
    mol_types[np.isin(aa.chain_type, ChainType.get_non_polymers())] = "ligand"
    hybrid = aa.chain_type == ChainType.DNA_RNA_HYBRID
    mol_types[hybrid & np.isin(aa.res_name, ["DA", "DC", "DG", "DT", "DI", "DN"])] = (
        "dna"
    )
    mol_types[hybrid & np.isin(aa.res_name, ["A", "C", "G", "U", "I", "N"])] = "rna"

    sources = {}
    if config.get("conformer_restraints_config") is not None:
        sources = _ligand_sources(input_spec, set(aa.chain_id[mol_types == "ligand"]))

    adapter = RF3Adapter(
        aa,
        atom_to_token_map=feats["atom_to_token_map"].detach().cpu().numpy(),
        ref_pos=feats["ref_pos"].detach().cpu().numpy(),
        ref_space_uid=feats["ref_space_uid"].detach().cpu().numpy(),
        mol_types=mol_types,
        ligand_mols=sources,
    )
    restraints = CombinedRestraints()
    restraints.setup(adapter, nbatch=int(pipeline_output["t"].shape[0]), config=config)
    return restraints
