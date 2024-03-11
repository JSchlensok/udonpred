# TODO remove unused imports
import datetime
import yaml
from datetime import datetime as dt
from pathlib import Path
from typing_extensions import Annotated

import h5py
import numpy as np
import polars as pl
import rich.progress as progress
import torch
import typer
from Bio import SeqIO

from src.models import FNN
from src.utils import embedding_dimensions

def write_caid_file(directory: Path, protein_id: str, scores: torch.tensor, sequence: str) -> None:
    with open(directory / (protein_id + ".caid"), "w+") as f:
        f.write(">" + protein_id + "\n")
        for i, (score, aa) in enumerate(zip(scores, sequence)):
            f.write(f"{i+1}\t{aa}\t{score:.3f}\n")

# TODO parametrize number of cores
# TODO try multithreading using joblib
def main(
    fasta_file: Annotated[Path, typer.Option("--input-fasta", "-i")],
    embedding_file: Annotated[Path, typer.Option("--embedding-file", "-e")] = None
):
    script_start_time = dt.now()

    # Load model
    device = "cpu"
    embedding_type = "prostt5"
    method_name = "FNN"
    embedding_dim = embedding_dimensions[embedding_type]

    model_dir = Path.cwd() / "model"
    model_config = yaml.safe_load((model_dir / "config.yml").open())
    
    map_location = torch.device("cpu") if not torch.cuda.is_available() else None
    model = FNN(n_features=embedding_dim, **params)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=map_location))

    # Prepare file output
    output_path = Path.cwd() / "disorder"
    output_path.mkdir(exist_ok=True, parents=True)

    sequences = {rec.id: rec.seq for rec in SeqIO.parse(fasta_file, "fasta")}
    execution_times = []

    with progress.Progress(
        *progress.Progress.get_default_columns(), progress.TimeElapsedColumn()
    ) as pbar, torch.no_grad():
        if embedding_file is None:
            # TODO generate embeddings
            pass

        embeddings = {id: torch.tensor(np.array(emb[()]), device=device) for id, emb in h5py.File(embedding_file).items() if id in sequences.keys()}
        overall_progress = pbar.add_task("Prediction progress", total=len(embeddings))

        for id, emb in embeddings.items():
            start_time = dt.now()
            emb = emb.clone().to(device=device, dtype=torch.float64)
            pred = model(emb)

            write_caid_file(output_path, id, pred, sequences[id])
            end_time = dt.now()
            execution_time = end_time - start_time
            execution_times.append({"sequence": id, "milliseconds": execution_time / datetime.timedelta(milliseconds=1)})
            
            pbar.advance(overall_progress)

        csv_lines = pl.from_records(execution_times).write_csv(file=None)
        with open("timings.csv", "w+") as f:
            f.write(f"# Running test, started {script_start_time:%c}\n")
            f.write(csv_lines)

if __name__ == "__main__":
    typer.run(main)
