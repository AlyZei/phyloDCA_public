from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path("/home/alya/phyloDCA")
ADABMDCA_ROOT = ROOT_DIR / "adabmDCApy"
if str(ADABMDCA_ROOT) not in sys.path:
    sys.path.insert(0, str(ADABMDCA_ROOT))

from adabmDCA.io import save_params


DEFAULT_TOKENS = "-ACDEFGHIKLMNPQRSTVWY"


def _is_int_token(x: str) -> bool:
    try:
        int(x)
        return True
    except ValueError:
        return False


def detect_param_format(path: Path) -> str:
    """
    Detect Potts parameter format from h/J lines.

    Returns:
        "numeric" if aa fields are integers
        "token"   if aa fields are alphabet tokens
    """
    saw_numeric = False
    saw_token = False

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "h":
                aa = parts[2]
            elif tag == "J":
                aa = parts[3]
            else:
                continue

            if _is_int_token(aa):
                saw_numeric = True
            else:
                saw_token = True

            if saw_numeric and saw_token:
                raise ValueError("Mixed numeric/token amino-acid labels found; file is inconsistent.")

    if saw_numeric:
        return "numeric"
    if saw_token:
        return "token"
    raise ValueError("Could not detect format: no valid h/J records found.")


def convert_numeric_to_token(in_path: Path, out_path: Path, tokens: str = DEFAULT_TOKENS) -> None:
    # 1) Parse FraZ numeric file into dense arrays (L,L,q,q) and (L,q)
    couplings = {}
    fields = {}
    q = len(tokens)

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "J":
                i, j = int(parts[1]), int(parts[2])
                a, b = int(parts[3]), int(parts[4])
                if not (0 <= a < q and 0 <= b < q):
                    raise ValueError(f"J indices out of range: ({a},{b}) for q={q}")
                couplings[(i, j, a, b)] = float(parts[5])
            elif parts[0] == "h":
                i, a = int(parts[1]), int(parts[2])
                if not (0 <= a < q):
                    raise ValueError(f"h index out of range: {a} for q={q}")
                fields[(i, a)] = float(parts[3])

    if not fields:
        raise ValueError(f"No h entries found in {in_path}")

    L = max(i for i, _ in fields.keys()) + 1
    h = np.zeros((L, q), dtype=np.float32)
    J_llqq = np.zeros((L, L, q, q), dtype=np.float32)

    for (i, a), v in fields.items():
        h[i, a] = v
    for (i, j, a, b), v in couplings.items():
        J_llqq[i, j, a, b] = v
        J_llqq[j, i, b, a] = v

    # 2) Build adabmDCA params dict with required keys/shapes
    params = {
        "bias": torch.tensor(h, dtype=torch.float32),
        # adabmDCA expects (L, q, L, q)
        "coupling_matrix": torch.tensor(J_llqq.transpose(0, 2, 1, 3), dtype=torch.float32),
    }

    # 3) Save through adabmDCA writer (ensures token alphabet mapping in output)
    save_params(str(out_path), params=params, tokens=tokens)


def convert_token_to_numeric(in_path: Path, out_path: Path, tokens: str = DEFAULT_TOKENS) -> None:
    token_to_idx = {t: i for i, t in enumerate(tokens)}
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "h":
                i = parts[1]
                aa = parts[2]
                if aa not in token_to_idx:
                    raise ValueError(f"Unexpected token in h line: '{aa}'")
                v = parts[3]
                fout.write(f"h {i} {token_to_idx[aa]} {v}\n")
            elif tag == "J":
                i, j = parts[1], parts[2]
                aa0, aa1 = parts[3], parts[4]
                if aa0 not in token_to_idx or aa1 not in token_to_idx:
                    raise ValueError(f"Unexpected token(s) in J line: '{aa0}', '{aa1}'")
                v = parts[5]
                fout.write(f"J {i} {j} {token_to_idx[aa0]} {token_to_idx[aa1]} {v}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Potts parameter files between numeric and token formats.")
    parser.add_argument("--input", required=True, help="Input parameter file path")
    parser.add_argument("--output", required=True, help="Output parameter file path")
    parser.add_argument(
        "--to",
        choices=["numeric", "token", "auto"],
        default="auto",
        help="Target format. 'auto' flips source format.",
    )
    parser.add_argument("--tokens", default=DEFAULT_TOKENS, help="Alphabet token order")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    src = detect_param_format(in_path)

    if args.to == "auto":
        dst = "token" if src == "numeric" else "numeric"
    else:
        dst = args.to

    if src == dst:
        raise ValueError(f"Input is already {src}; requested same target format.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if src == "numeric" and dst == "token":
        convert_numeric_to_token(in_path, out_path, tokens=args.tokens)
    elif src == "token" and dst == "numeric":
        convert_token_to_numeric(in_path, out_path, tokens=args.tokens)
    else:
        raise RuntimeError(f"Unsupported conversion path: {src} -> {dst}")

    print(f"Converted {in_path} ({src}) -> {out_path} ({dst})")


if __name__ == "__main__":
    main()
