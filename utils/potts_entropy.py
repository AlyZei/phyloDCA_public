"""
Fast entropy estimation for adabmDCApy Potts models (L~200, q=21)
with fixed couplings and varying fields.

Mathematical basis
------------------
The Potts Boltzmann distribution is:
    p(x) = exp(-E(x)) / Z,  E(x) = -sum_i h_i(x_i) - sum_{i<j} J_ij(x_i, x_j)

The entropy is:
    S = -sum_x p(x) log p(x) = log Z + <E>_p

Two complementary strategies are implemented:

1. **AIS-based entropy** (Annealed Importance Sampling)
   - Bridges a factored (independent-site) reference model at lambda=0
     to the full model at lambda=1 via a schedule of lambda values.
   - Estimates log Z_target - log Z_ref via importance weights.
   - log Z_ref is computed exactly (sum of per-site log-sum-exp over fields only).
   - <E> is estimated from the final AIS samples.
   - Cost: O(n_chains * n_steps * L * q^2) — fully vectorised on CPU with torch.
   - Correct in the limit of many chains/steps; unbiased estimator of log Z.

2. **Thermodynamic Integration over fields** (fast for field variants)
   - Use the identity:  d log Z / d h_i(a) = <delta_{x_i, a}>_p = f_i(a)
   - So: log Z(h') - log Z(h) = integral_0^1 sum_{i,a} (h'_ia - h_ia) * <f_ia>_{lambda} dlambda
   - where h_lambda = h + lambda*(h' - h).
   - If you already have an MCMC estimate of <E> and log Z for the *reference* model
     (h, J), you can cheaply get S for nearby field variants by integrating
     the change in log Z using short MCMC runs at each lambda step.
   - When the field change is small, a single linear interpolation with a handful
     of Gauss-Legendre quadrature points suffices.

Both methods use adabmDCApy's own `compute_energy` and Gibbs sampler internals.
"""

import math
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from adabmDCA.statmech import compute_energy
from adabmDCA.statmech import _update_weights_AIS
from adabmDCA.sampling import gibbs_sampling   # one Gibbs sweep over all sites


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_z_independent(bias: torch.Tensor) -> float:
    """
    Exact log Z for the factored (zero-coupling) reference model.
    log Z_0 = sum_i log( sum_a exp(h_i(a)) )

    Args:
        bias: (L, q) tensor of fields.
    Returns:
        float
    """
    return bias.logsumexp(dim=-1).sum().item()


def _gibbs_sweep_batch(
    chains: torch.Tensor,       # (B, L) int64 categorical
    params: Dict[str, torch.Tensor],
    n_sweeps: int = 1,
) -> torch.Tensor:
    """
    Run n_sweeps full Gibbs sweeps on a batch of chains.
    Wraps adabmDCApy's gibbs_sampling which expects one-hot tensors.
    """
    L, q = params["bias"].shape
    # one-hot encode
    x_oh = torch.zeros(chains.shape[0], L, q, dtype=params["bias"].dtype,
                        device=chains.device)
    x_oh.scatter_(2, chains.unsqueeze(-1), 1.0)
    for _ in range(n_sweeps):
        x_oh = gibbs_sampling(chains=x_oh, params=params, nsweeps=1)
    # decode back to integer categories
    return x_oh.argmax(dim=-1)


def _mean_energy_from_chains(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    batch_size: int = 512,
) -> float:
    """
    Estimate <E> = mean DCA energy over chains. Processes in mini-batches
    to avoid OOM on large chain sets.
    """
    L, q = params["bias"].shape
    total, n = 0.0, 0
    for start in range(0, len(chains), batch_size):
        chunk = chains[start : start + batch_size]
        x_oh = torch.zeros(len(chunk), L, q, dtype=params["bias"].dtype,
                            device=chains.device)
        x_oh.scatter_(2, chunk.unsqueeze(-1), 1.0)
        e = compute_energy(x_oh, params)
        total += e.sum().item()
        n += len(chunk)
    return total / n


# ---------------------------------------------------------------------------
# Strategy 1: AIS — full entropy estimate for a single model
# ---------------------------------------------------------------------------

def entropy_ais(
    params: Dict[str, torch.Tensor],
    n_chains: int = 2000,
    n_beta_steps: int = 200,
    n_sweeps_per_step: int = 1,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Estimate entropy S of a Potts model via Annealed Importance Sampling.

    The annealing path interpolates linearly between:
      - p_0(x) = prod_i softmax(h_i)  (independent sites, exact log Z_0)
      - p_1(x) = true Potts model

    The intermediate energy at inverse-temperature beta is:
      E_beta(x) = -beta * [sum_i h_i(x_i) + sum_{i<j} J_ij(x_i,x_j)]
                  - (1-beta) * sum_i h_i(x_i)
                = -sum_i h_i(x_i) - beta * sum_{i<j} J_ij(x_i,x_j)

    i.e. we only anneal the couplings, keeping full fields throughout.
    This gives a better-conditioned path for protein families where
    fields dominate.

    AIS log-weight accumulator:
      log w += log p_{beta_{k+1}}(x) - log p_{beta_k}(x)
             = -(beta_{k+1} - beta_k) * E_coupling(x)

    where E_coupling(x) = -sum_{i<j} J_ij(x_i, x_j).

    Then: log Z_1 - log Z_0 = log mean_chains(exp(log w))   [log-sum-exp trick]
    And:  S = log Z_1 + <E>_{p_1}

    Args:
        params:           adabmDCApy params dict with "bias" (L,q) and
                          "coupling_matrix" (L,q,L,q).
        n_chains:         Number of parallel AIS chains.
        n_beta_steps:     Number of annealing steps (more = more accurate).
        n_sweeps_per_step: Gibbs sweeps between beta increments.
        device:           torch device (cpu recommended for L=200).
        seed:             Random seed.

    Returns:
        (entropy_estimate, log_Z_estimate)
    """
    if seed is not None:
        torch.manual_seed(seed)

    bias = params["bias"].to(device)           # (L, q)
    J    = params["coupling_matrix"].to(device) # (L, q, L, q)
    L, q = bias.shape

    params_device = {"bias": bias, "coupling_matrix": J}

    # --- initialise chains by sampling from p_0 (independent sites) ---
    probs_0 = torch.softmax(bias, dim=-1)  # (L, q)
    chains = torch.multinomial(
        probs_0.view(L, q).expand(n_chains, -1, -1).reshape(n_chains * L, q),
        num_samples=1,
    ).view(n_chains, L)  # (B, L)

    betas = torch.linspace(0, 1, n_beta_steps + 1, device=device)
    log_weights = torch.zeros(n_chains, device=device)

    for k in range(n_beta_steps):
        # Use adabmDCA's exact AIS weight update: log w += E_prev - E_curr.
        params_prev = {
            "bias": bias,
            "coupling_matrix": betas[k] * J,
        }
        params_curr = {
            "bias": bias,
            "coupling_matrix": betas[k + 1] * J,
        }
        x_oh = torch.zeros(n_chains, L, q, dtype=bias.dtype, device=device)
        x_oh.scatter_(2, chains.unsqueeze(-1), 1.0)
        log_weights = _update_weights_AIS(params_prev, params_curr, x_oh, log_weights)

        # Gibbs sweeps at beta_{k+1}
        x_oh = torch.zeros(n_chains, L, q, dtype=bias.dtype, device=device)
        x_oh.scatter_(2, chains.unsqueeze(-1), 1.0)
        for _ in range(n_sweeps_per_step):
            x_oh = gibbs_sampling(chains=x_oh, params=params_curr, nsweeps=1)
        chains = x_oh.argmax(dim=-1)

    # --- log Z estimate ---
    log_Z_0 = _log_z_independent(bias)
    # log mean exp(log_weights) with numerical stability
    log_Z_delta = torch.logsumexp(log_weights, dim=0).item() - math.log(n_chains)
    log_Z = log_Z_0 + log_Z_delta

    # --- <E> from final chains (which approximate p_1) ---
    mean_E = _mean_energy_from_chains(chains, params_device)

    entropy = log_Z + mean_E  # S = log Z + <E>  (energies are negative, so S > 0)
    return entropy, log_Z


# ---------------------------------------------------------------------------
# Strategy 2: Thermodynamic Integration over fields (fast field variants)
# ---------------------------------------------------------------------------

def entropy_field_variant_ti(
    params_ref: Dict[str, torch.Tensor],
    log_Z_ref: float,
    mean_E_ref: float,
    chains_ref: torch.Tensor,
    new_bias: torch.Tensor,
    n_quad_points: int = 5,
    n_sweeps_per_point: int = 5,
    n_final_sweeps: Optional[int] = None,
    n_chains: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """
    Fast entropy estimate for a field variant model using thermodynamic integration.

    Given a reference model (h, J) with known log Z_ref and <E>_ref,
    compute log Z and S for the variant model (h', J) — same couplings.

    The identity is:
        d log Z / d lambda = sum_{i,a} Delta_h[i,a] * <f_ia>_{lambda}
    where Delta_h = h' - h and f_ia = <delta_{x_i,a}> is the single-site
    frequency under the interpolated model h_lambda = h + lambda * Delta_h.

    Integration via Gauss-Legendre quadrature over lambda in [0,1].

    For each quadrature point lambda_k with weight w_k:
      1. Set h_lambda = h + lambda_k * Delta_h
      2. Run short Gibbs chain (warm-started from reference chains)
      3. Estimate fi = single-site frequencies
      4. Accumulate: delta_logZ += w_k * sum_{i,a} Delta_h[i,a] * fi[i,a]

    Then:
        log Z_new = log Z_ref + delta_logZ
        <E>_new   from final chains at lambda=1
        S_new     = log Z_new + <E>_new

    Args:
        params_ref:       Reference params (bias, coupling_matrix).
        log_Z_ref:        log Z of reference model (from AIS or prior run).
        mean_E_ref:       <E> of reference model (for sanity check).
        chains_ref:       Equilibrium chains from reference model, shape (B, L) int64.
        new_bias:         New bias tensor (L, q) for the variant.
        n_quad_points:    Gauss-Legendre nodes in [0,1].
        n_sweeps_per_point: Gibbs sweeps at each quadrature point.
        n_final_sweeps:   Extra Gibbs sweeps at lambda=1 before measuring <E>.
                  If None, defaults to n_sweeps_per_point.
        n_chains:         If not None, subsample chains_ref to this size.
        device:           torch device.

    Returns:
        (entropy_new, log_Z_new)
    """
    bias_ref = params_ref["bias"].to(device)
    J = params_ref["coupling_matrix"].to(device)
    L, q = bias_ref.shape
    h_new = new_bias.to(device)
    delta_h = h_new - bias_ref  # (L, q)

    chains = chains_ref.to(device)
    if n_chains is not None and n_chains < len(chains):
        idx = torch.randperm(len(chains))[:n_chains]
        chains = chains[idx]
    B = len(chains)
    if n_final_sweeps is None:
        n_final_sweeps = n_sweeps_per_point

    # Gauss-Legendre nodes and weights on [0,1]
    nodes, weights = np.polynomial.legendre.leggauss(n_quad_points)
    # map from [-1,1] to [0,1]
    lambdas = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights  # Jacobian

    delta_logZ = 0.0

    for lam, w in zip(lambdas, weights):
        h_lam = bias_ref + lam * delta_h
        params_lam = {"bias": h_lam, "coupling_matrix": J}

        # Gibbs sweeps (warm-started)
        x_oh = torch.zeros(B, L, q, dtype=bias_ref.dtype, device=device)
        x_oh.scatter_(2, chains.unsqueeze(-1), 1.0)
        for _ in range(n_sweeps_per_point):
            x_oh = gibbs_sampling(chains=x_oh, params=params_lam, nsweeps=1)
        chains = x_oh.argmax(dim=-1)

        # Single-site frequencies fi[i, a] = mean over chains of delta_{x_i, a}
        fi = x_oh.mean(dim=0)  # (L, q)

        # Integrand: sum_{i,a} delta_h[i,a] * fi[i,a]
        integrand = (delta_h * fi).sum().item()
        delta_logZ += w * integrand

    log_Z_new = log_Z_ref + delta_logZ

    # Gauss-Legendre nodes exclude endpoints; equilibrate explicitly at lambda=1.
    params_new = {"bias": h_new, "coupling_matrix": J}
    if n_final_sweeps > 0:
        x_oh = torch.zeros(B, L, q, dtype=bias_ref.dtype, device=device)
        x_oh.scatter_(2, chains.unsqueeze(-1), 1.0)
        for _ in range(n_final_sweeps):
            x_oh = gibbs_sampling(chains=x_oh, params=params_new, nsweeps=1)
        chains = x_oh.argmax(dim=-1)

    # <E> at new model from lambda=1 chains.
    mean_E_new = _mean_energy_from_chains(chains, params_new)

    entropy_new = log_Z_new + mean_E_new
    return entropy_new, log_Z_new


# ---------------------------------------------------------------------------
# Convenience: batch field variants
# ---------------------------------------------------------------------------

def entropy_batch_field_variants(
    params_ref: Dict[str, torch.Tensor],
    log_Z_ref: float,
    mean_E_ref: float,
    chains_ref: torch.Tensor,
    variant_biases: List[torch.Tensor],
    n_quad_points: int = 5,
    n_sweeps_per_point: int = 5,
    n_chains: int = 2000,
    device: torch.device = torch.device("cpu"),
) -> List[Tuple[float, float]]:
    """
    Estimate entropy for a list of field variants, all sharing the same couplings.

    Each variant is computed independently (using TI from the reference).
    If variants are close to each other, consider chaining them (TI from previous
    variant instead of reference), which is trivially doable by replacing
    params_ref/log_Z_ref/chains_ref with the previous result.

    Args:
        params_ref:        Reference Potts model params.
        log_Z_ref:         log Z of reference model.
        mean_E_ref:        <E> of reference model.
        chains_ref:        Reference equilibrium chains (B, L) int64.
        variant_biases:    List of new bias tensors, each (L, q).
        n_quad_points:     GL quadrature points for TI.
        n_sweeps_per_point: Gibbs sweeps per quadrature point.
        n_chains:          Number of chains to use (subsampled from chains_ref).
        device:            Torch device.

    Returns:
        List of (entropy, log_Z) tuples, one per variant.
    """
    results = []
    for i, h_new in enumerate(variant_biases):
        S, logZ = entropy_field_variant_ti(
            params_ref=params_ref,
            log_Z_ref=log_Z_ref,
            mean_E_ref=mean_E_ref,
            chains_ref=chains_ref,
            new_bias=h_new,
            n_quad_points=n_quad_points,
            n_sweeps_per_point=n_sweeps_per_point,
            n_chains=n_chains,
            device=device,
        )
        results.append((S, logZ))
        print(f"  Variant {i+1}/{len(variant_biases)}: S = {S:.4f}, logZ = {logZ:.4f}")
    return results


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Typical workflow:

    Step 0. Load your model.
        from adabmDCA.io import load_params
        params = load_params("my_model.pt")   # or however you saved it

    Step 1. Get a baseline entropy estimate for the reference model (h, J).
        This is the expensive step, but you only do it once.
        S_ref, logZ_ref = entropy_ais(
            params,
            n_chains=4000,
            n_beta_steps=500,
            n_sweeps_per_step=1,
        )
        # Also keep the equilibrium chains for TI warm-starting.
        # Either reuse AIS final chains (returned by a modified version),
        # or run a short Gibbs burn-in:

        from adabmDCA.sampling import gibbs_sampling
        L, q = params["bias"].shape
        x = torch.randint(q, (4000, L))
        x_oh = torch.zeros(4000, L, q)
        x_oh.scatter_(2, x.unsqueeze(-1), 1.0)
        for _ in range(200):
            x_oh = gibbs_sampling(chains=x_oh, params=params, nsweeps=1)
        chains_ref = x_oh.argmax(dim=-1)
        mean_E_ref = _mean_energy_from_chains(chains_ref, params)

    Step 2. For each field variant, use TI (very fast):
        new_bias = params["bias"] + 0.1 * torch.randn_like(params["bias"])
        S_variant, logZ_variant = entropy_field_variant_ti(
            params_ref=params,
            log_Z_ref=logZ_ref,
            mean_E_ref=mean_E_ref,
            chains_ref=chains_ref,
            new_bias=new_bias,
            n_quad_points=5,       # 5 is usually enough for small delta_h
            n_sweeps_per_point=10,
        )

    Complexity notes (L=200, q=21, B=2000 chains):
    - AIS (500 steps, 1 sweep): ~500 Gibbs sweeps on 2000 chains ≈ seconds on CPU
    - TI per variant (5 GL pts, 10 sweeps): ~50 Gibbs sweeps ≈ <1 second on CPU
    """
    print("Running self-test with a small synthetic model (L=10, q=5)...")
    torch.manual_seed(42)
    L, q = 10, 5
    bias = 0.5 * torch.randn(L, q)
    J = torch.zeros(L, q, L, q)
    # sparse random couplings
    for i in range(L):
        for j in range(i+1, L):
            if torch.rand(1).item() < 0.3:
                v = 0.3 * torch.randn(q, q)
                J[i, :, j, :] = v
                J[j, :, i, :] = v.T
    params = {"bias": bias, "coupling_matrix": J}

    print("  AIS entropy estimation...")
    S_ais, logZ_ais = entropy_ais(params, n_chains=500, n_beta_steps=100, seed=0)
    print(f"  AIS:  S = {S_ais:.4f},  logZ = {logZ_ais:.4f}")

    # Warm chains for TI
    x = torch.randint(q, (500, L))
    x_oh = torch.zeros(500, L, q)
    x_oh.scatter_(2, x.unsqueeze(-1), 1.0)
    for _ in range(50):
        x_oh = gibbs_sampling(chains=x_oh, params=params, nsweeps=1)
    chains_ref = x_oh.argmax(dim=-1)
    mean_E_ref = _mean_energy_from_chains(chains_ref, params)

    print("  TI field-variant entropy estimation...")
    new_bias = bias + 0.2 * torch.randn(L, q)
    S_ti, logZ_ti = entropy_field_variant_ti(
        params_ref=params,
        log_Z_ref=logZ_ais,
        mean_E_ref=mean_E_ref,
        chains_ref=chains_ref,
        new_bias=new_bias,
        n_quad_points=7,
        n_sweeps_per_point=10,
    )
    print(f"  TI variant:  S = {S_ti:.4f},  logZ = {logZ_ti:.4f}")

    # Ground-truth for small model: AIS on the variant directly
    params_new = {"bias": new_bias, "coupling_matrix": J}
    S_ais_v, logZ_ais_v = entropy_ais(params_new, n_chains=500, n_beta_steps=100, seed=1)
    print(f"  AIS variant: S = {S_ais_v:.4f},  logZ = {logZ_ais_v:.4f}  (ground truth)")
    print(f"  TI error: dS = {abs(S_ti - S_ais_v):.4f}, d logZ = {abs(logZ_ti - logZ_ais_v):.4f}")
    print("Done.")
