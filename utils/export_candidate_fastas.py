"""
Export candidate sequences (GT/MAP/Consensus/DCA top10/Yang top10) to FASTA files.

One FASTA file is written per mu value (typically the mu interval used in Figure 5).
Within each file, entries are written in this order for i = 1..5:
  - GTi
  - MAPi
  - Consi
  - DCAi_1 ... DCAi_10
  - Yangi_1 ... Yangi_10

Sequences are exported as amino-acid strings using the mapping defined in
utils.toolsForTreesAndMSAs.int_to_amino_acid_seq.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.PottsEnergies import energy_of_msa
from utils.paper_figures_export import select_topN
from utils.toolsForTreesAndMSAs import int_to_amino_acid_seq
from utils.utils import get_all_file_paths


def _as_int_seq(seq: Sequence[Any]) -> np.ndarray:
    arr = np.asarray(seq)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D sequence, got shape {arr.shape}")
    return arr.astype(int)


def _load_consensus_dict(
    consensus_directory: str,
    sequences: Sequence[str],
    mu_values: Sequence[float],
) -> Dict[Tuple[str, float], np.ndarray]:
    consensus_dict: Dict[Tuple[str, float], np.ndarray] = {}
    for file in get_all_file_paths(consensus_directory):
        if "reweighted" in file:
            continue
        filename = os.path.basename(file)
        for seq in sequences:
            if seq not in filename:
                continue
            for mu in mu_values:
                # keep same matching convention as paper_figures_export.py
                mu_pattern = f"_mu{float(mu):.1f}_"
                if mu_pattern in filename:
                    consensus_dict[(seq, float(mu))] = np.loadtxt(file, dtype=int)
                    break
    return consensus_dict


def _fasta_write(handle, header: str, seq_int: np.ndarray) -> None:
    handle.write(f">{header}\n")
    handle.write(f"{int_to_amino_acid_seq(_as_int_seq(seq_int).tolist())}\n")


def _maybe_remove_gaps(seq_int: np.ndarray, remove_gaps: bool) -> np.ndarray:
    """Remove gap token (0) from an integer-encoded sequence if requested."""
    seq = _as_int_seq(seq_int)
    if not remove_gaps:
        return seq
    return seq[seq != 0]


def export_candidate_fastas_for_fig5(
    output_dir: str,
    sequences: Sequence[str],
    mu_values_reduced: Sequence[float],
    GT_sequences: Dict[str, Sequence[int]],
    msa_folder: str,
    posterior_folder: str,
    consensus_directory: str,
    fields_: np.ndarray,
    couplings: np.ndarray,
    M: int,
    T: float,
    data_prefix: str = "DBD",
    energy_keep_pct: Optional[float] = None,
    topN: int = 10,
    n_roots: int = 5,
    remove_gaps: bool = False,
) -> List[str]:
    """
    Write one FASTA per mu with GT/MAP/Consensus/top candidates for the first n_roots sequences.

    Returns:
        list of written FASTA file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    selected_roots = list(sequences)[:n_roots]
    if len(selected_roots) < n_roots:
        raise ValueError(f"Requested n_roots={n_roots}, but got only {len(selected_roots)} sequences")

    mu_values = [float(m) for m in mu_values_reduced]
    consensus_dict = _load_consensus_dict(consensus_directory, selected_roots, mu_values)

    output_files: List[str] = []

    for mu in mu_values:
        suffix = "_nogaps" if remove_gaps else ""
        out_path = os.path.join(output_dir, f"candidate_sequences_mu{mu:g}{suffix}.fasta")
        with open(out_path, "w", encoding="utf-8") as f:
            for i, root in enumerate(selected_roots, start=1):
                gt_seq = _as_int_seq(GT_sequences[root])
                _fasta_write(f, f"GT{i}", _maybe_remove_gaps(gt_seq, remove_gaps))

                posterior_path = os.path.join(
                    posterior_folder,
                    f"{data_prefix}_{root}_mu{mu}_ancestral_probability",
                )
                if not os.path.exists(posterior_path):
                    raise FileNotFoundError(f"Missing posterior file: {posterior_path}")

                posterior = np.loadtxt(posterior_path)
                map_seq = np.argmax(posterior, axis=1).astype(int)
                _fasta_write(f, f"MAP{i}", _maybe_remove_gaps(map_seq, remove_gaps))

                cons_key = (root, float(mu))
                if cons_key not in consensus_dict:
                    raise FileNotFoundError(
                        f"Missing consensus for sequence={root}, mu={mu}. "
                        f"Check files in {consensus_directory}."
                    )
                cons_seq = _as_int_seq(consensus_dict[cons_key])
                _fasta_write(f, f"Cons{i}", _maybe_remove_gaps(cons_seq, remove_gaps))

                reshuffled_path = os.path.join(
                    msa_folder,
                    f"{root}_mu={mu}_depth=None_shuffled_M={M}_T={T}",
                )
                if not os.path.exists(reshuffled_path):
                    raise FileNotFoundError(f"Missing reshuffled MSA: {reshuffled_path}")
                dca_reshuffled = np.loadtxt(reshuffled_path, dtype=int)
                if dca_reshuffled.ndim == 1:
                    dca_reshuffled = dca_reshuffled.reshape(1, -1)

                energies_reshuffled = energy_of_msa(dca_reshuffled, fields_, couplings)
                dca_idx = select_topN(
                    dca_reshuffled,
                    energies_reshuffled,
                    posterior=posterior,
                    scoring="yang",
                    percentage=energy_keep_pct,
                    topN=topN,
                )
                for rank, idx in enumerate(dca_idx[:topN], start=1):
                    _fasta_write(
                        f,
                        f"DCA{i}_{rank}",
                        _maybe_remove_gaps(_as_int_seq(dca_reshuffled[int(idx)]), remove_gaps),
                    )

                posterior_msa_path = os.path.join(
                    msa_folder,
                    f"{root}_mu={mu}_depth=None_M={M}",
                )
                if not os.path.exists(posterior_msa_path):
                    raise FileNotFoundError(f"Missing posterior-sampled MSA: {posterior_msa_path}")
                posterior_msa = np.loadtxt(posterior_msa_path, dtype=int)
                if posterior_msa.ndim == 1:
                    posterior_msa = posterior_msa.reshape(1, -1)

                energies_posterior = energy_of_msa(posterior_msa, fields_, couplings)
                yang_idx = select_topN(
                    posterior_msa,
                    energies_posterior,
                    posterior=posterior,
                    scoring="yang",
                    percentage=None,
                    topN=topN,
                )
                for rank, idx in enumerate(yang_idx[:topN], start=1):
                    _fasta_write(
                        f,
                        f"Yang{i}_{rank}",
                        _maybe_remove_gaps(_as_int_seq(posterior_msa[int(idx)]), remove_gaps),
                    )

        output_files.append(out_path)

    return output_files


def export_candidate_fastas_from_export_config(
    export_config: Dict[str, Any],
    output_dir: str,
    topN: int = 10,
    n_roots: int = 5,
    remove_gaps: bool = False,
) -> List[str]:
    """Convenience wrapper using the same config dictionary used by paper figure exports."""
    mu_values_reduced = export_config.get("mu_values_reduced")
    if mu_values_reduced is None:
        mu_values_reduced = [mu for mu in export_config["mu_values"] if 1.0 <= float(mu) <= 100.0]

    return export_candidate_fastas_for_fig5(
        output_dir=output_dir,
        sequences=export_config["sequences"],
        mu_values_reduced=mu_values_reduced,
        GT_sequences=export_config["GT_sequences"],
        msa_folder=export_config["msa_save_folder"],
        posterior_folder=export_config["ancestral_save_folder"],
        consensus_directory=export_config["consensus_directory"],
        fields_=export_config["fields_"],
        couplings=export_config["couplings"],
        M=int(export_config["M"]),
        T=float(export_config["T"]),
        data_prefix=export_config.get("data_prefix", "DBD"),
        energy_keep_pct=export_config.get("energy_keep_pct", None),
        topN=topN,
        n_roots=n_roots,
        remove_gaps=remove_gaps,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export candidate FASTA files (one file per mu) for Figure-5 candidate sets."
    )
    parser.add_argument("--output-dir", required=True, help="Directory where FASTA files will be written")
    parser.add_argument("--config-npz", required=True, help=(
        "Path to a .npz file containing keys: sequences, mu_values_reduced, GT_sequences, "
        "fields_, couplings, msa_save_folder, ancestral_save_folder, consensus_directory, M, T, data_prefix"
    ))
    parser.add_argument("--topN", type=int, default=10, help="Number of DCA/Yang candidates per root")
    parser.add_argument("--n-roots", type=int, default=5, help="Number of roots to export (default: 5)")
    parser.add_argument(
        "--remove-gaps",
        action="store_true",
        help="If set, remove gap token '-' from all exported sequences and append '_nogaps' to filenames.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    payload = np.load(args.config_npz, allow_pickle=True)

    sequences = payload["sequences"].tolist()
    mu_values_reduced = payload["mu_values_reduced"].tolist()
    GT_sequences = payload["GT_sequences"].item()
    fields_ = payload["fields_"]
    couplings = payload["couplings"]

    output_files = export_candidate_fastas_for_fig5(
        output_dir=args.output_dir,
        sequences=sequences,
        mu_values_reduced=mu_values_reduced,
        GT_sequences=GT_sequences,
        msa_folder=str(payload["msa_save_folder"].item()),
        posterior_folder=str(payload["ancestral_save_folder"].item()),
        consensus_directory=str(payload["consensus_directory"].item()),
        fields_=fields_,
        couplings=couplings,
        M=int(payload["M"].item()),
        T=float(payload["T"].item()),
        data_prefix=str(payload["data_prefix"].item()) if "data_prefix" in payload else "DBD",
        energy_keep_pct=float(payload["energy_keep_pct"].item()) if "energy_keep_pct" in payload else None,
        topN=args.topN,
        n_roots=args.n_roots,
        remove_gaps=args.remove_gaps,
    )

    print("Written files:")
    for p in output_files:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
