from pathlib import Path
from collections import OrderedDict
import subprocess
import tempfile

def read_fasta(path):
    seqs = OrderedDict()
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
                seqs[header] = []
            else:
                seqs[header].append(line)
    return {h: "".join(s) for h, s in seqs.items()}


def write_fasta(seqs, path):
    with open(path, "w") as f:
        for h, s in seqs.items():
            f.write(f">{h}\n")
            f.write(s + "\n")


def mask_gaps(seq, gap_char="-", mask_char="X"):
    return seq.replace(gap_char, mask_char)


def restore_gaps(masked_seq, original_seq, gap_char="-", mask_char="X"):
    # Restore gaps exactly where they were in the original alignment
    restored = []
    for m, o in zip(masked_seq, original_seq):
        restored.append(gap_char if o == gap_char else m)
    return "".join(restored)

def run_cdhit(input_fasta, output_fasta, identity):
    cmd = [
        "cd-hit",
        "-i", str(input_fasta),
        "-o", str(output_fasta),
        "-c", str(identity),
        "-n", "5",          # safe default for ~0.9–1.0 identity
        "-d", "0"           # keep full headers
    ]
    subprocess.run(cmd, check=True)

def cdhit_on_alignment(
    aligned_fasta,
    identity=0.97,
    gap_char="-",
    mask_char="X",
    output_fasta="clustered_alignment.fasta"
):
    aligned_fasta = Path(aligned_fasta)
    output_fasta = Path(output_fasta)

    # Read original alignment
    aligned_seqs = read_fasta(aligned_fasta)

    # Sanity check: all sequences same length
    lengths = {len(s) for s in aligned_seqs.values()}
    if len(lengths) != 1:
        raise ValueError("Input FASTA is not a proper alignment (lengths differ)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        masked_fasta = tmpdir / "masked.fasta"
        cdhit_out = tmpdir / "cdhit_out.fasta"

        # Mask gaps
        masked_seqs = {
            h: mask_gaps(s, gap_char, mask_char)
            for h, s in aligned_seqs.items()
        }
        write_fasta(masked_seqs, masked_fasta)

        # Run CD-HIT
        run_cdhit(masked_fasta, cdhit_out, identity)

        # Read CD-HIT representatives
        reps = read_fasta(cdhit_out)

    # Restore gaps for representatives
    restored = OrderedDict()
    for h, masked_seq in reps.items():
        original_seq = aligned_seqs[h]
        restored[h] = restore_gaps(
            masked_seq, original_seq, gap_char, mask_char
        )

    # Write final alignment
    write_fasta(restored, output_fasta)

    return restored

test_alignment = '/home/alya/phyloDCA/alignments_etc/PF13354_noinsert_max19gaps_nodupl_noclose_BetaLact.faa'

if __name__ =="__main__":
    clustered = cdhit_on_alignment(
    aligned_fasta=test_alignment,
    identity=0.97,
    output_fasta="/home/alya/phyloDCA/alignments_etc/PF13354_noinsert_max19gaps_nodupl_noclose_BetaLact_0.97.fa"
    )
    print(f"Final number of sequences: {len(clustered)}")
