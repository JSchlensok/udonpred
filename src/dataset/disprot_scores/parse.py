from collections import namedtuple
from pathlib import Path

from tqdm import tqdm


AnnotatedSequence = namedtuple("AnnotatedSequence", ["sequence", "annotations"])
annotation_map = {'0': 0, '1': 1, '-': None}

def read_score_fasta(fasta_file: Path, show_progress: bool = False) -> dict[str, ]:
    annotations = {}
    lines = [line.strip() for line in fasta_file.open().readlines()]
    indices = range(0, len(lines), 3)
    if show_progress:
        indices = tqdm(indices)
    
    for i in range(0, len(lines), 3):
        protein_id = lines[i].replace('>', '')
        sequence = lines[i+1]
        annotation = [annotation_map[x] for x in lines[i+2]]
        annotations[protein_id] = AnnotatedSequence(sequence, annotation)
    return annotations
