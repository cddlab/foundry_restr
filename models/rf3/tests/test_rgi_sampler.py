"""RGI hook ordering, schedule semantics, and inference-mode optimization."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from rf3.diffusion_samplers.inference_sampler import SampleDiffusion


def make_sampler(monkeypatch):
    sampler = SampleDiffusion(
        num_timesteps=3,
        min_t=0,
        max_t=1,
        sigma_data=16,
        s_min=0.001,
        s_max=1,
        p=7,
        gamma_0=0.8,
        gamma_min=0,
        noise_scale=0,
        step_scale=1,
        solver="af3",
    )
    monkeypatch.setattr(
        sampler,
        "_construct_inference_noise_schedule",
        lambda device: torch.tensor([4.0, 2.0, 0.0], device=device),
    )
    monkeypatch.setattr(
        "rf3.diffusion_samplers.inference_sampler.centre_random_augmentation",
        lambda coords, exists, scale: coords,
    )
    return sampler


def sample(sampler, restraints=None):
    initial = torch.tensor([[[0.0, 0, 0], [2.0, 0, 0]]]).repeat(2, 1, 1)
    return sampler.sample_diffusion_like_af3(
        S_inputs_I=torch.zeros(2, 1),
        S_trunk_I=torch.zeros(2, 1),
        Z_trunk_II=torch.zeros(2, 2, 1),
        f={"ref_element": torch.zeros(2, 128)},
        diffusion_module=lambda **kwargs: initial.clone(),
        diffusion_batch_size=2,
        coord_atom_lvl_to_be_noised=torch.zeros_like(initial),
        restraints=restraints,
    )


def test_hook_precedes_euler_and_uses_pre_churn_sigma(monkeypatch):
    sampler = make_sampler(monkeypatch)
    restraints = Mock()
    restraints.minimize.side_effect = lambda coords, step, sigma: coords + 3
    result = sample(sampler, restraints)
    assert [call.args[1] for call in restraints.minimize.call_args_list] == [0, 1]
    assert [float(call.args[2]) for call in restraints.minimize.call_args_list] == [
        4,
        2,
    ]
    assert float(result["t_hats"][0]) == pytest.approx(7.2)
    expected = torch.tensor([[[3.0, 3, 3], [5.0, 3, 3]]]).repeat(2, 1, 1)
    torch.testing.assert_close(result["X_L"], expected)
    assert restraints.finalize.call_args.args[0] is result["X_L"]
    assert restraints.finalize.call_args.args[1] == 1


def test_absent_and_empty_restraints_preserve_sampling(monkeypatch):
    from rgi_toolkit.combined import CombinedRestraints

    class Adapter:
        def iter_atoms(self):
            return iter(())

    restraints = CombinedRestraints()
    restraints.setup(Adapter(), nbatch=2, config={})
    sampler = make_sampler(monkeypatch)
    torch.manual_seed(13)
    baseline = sample(sampler)
    torch.manual_seed(13)
    empty = sample(sampler, restraints)
    torch.testing.assert_close(baseline["X_L"], empty["X_L"], rtol=0, atol=0)


def test_real_optimizer_in_inference_mode_and_step_window(monkeypatch):
    from rgi_toolkit.atom_context import AtomRecord
    from rgi_toolkit.combined import CombinedRestraints

    class Adapter:
        def iter_atoms(self):
            yield AtomRecord("A", 1, 0)
            yield AtomRecord("B", 1, 1)

    restraints = CombinedRestraints()
    restraints.setup(
        Adapter(),
        nbatch=2,
        config={
            "gpu": False,
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "harmonic": {"target_distance": 8},
                    "start_step": 1,
                    "stop_step": 1,
                }
            ],
        },
    )
    with torch.inference_mode():
        result = sample(make_sampler(monkeypatch), restraints)
    first = result["X_denoised_L_traj"][0].numpy()
    final = result["X_L"].numpy()
    np.testing.assert_allclose(np.linalg.norm(first[:, 1] - first[:, 0], axis=-1), 2)
    np.testing.assert_allclose(
        np.linalg.norm(final[:, 1] - final[:, 0], axis=-1), 8, atol=1e-4
    )
