import datetime
import logging
import re
from collections import OrderedDict
from datetime import datetime as dt
from pathlib import Path
from typing_extensions import Annotated

import h5py
import numpy as np
import polars as pl
import rich.progress as progress
import torch
import typer
import yaml
from Bio import SeqIO
from transformers import T5Tokenizer, T5EncoderModel

from src.models import FNN, CNN
from src.utils import embedding_dimensions

def generate_caid_format(protein_id: str, scores: torch.tensor, sequence: str) -> None:
    lines = []
    lines.append(">" + protein_id + "\n")
    for i, (score, aa) in enumerate(zip(scores, sequence)):
        lines.append(f"{i+1}\t{aa}\t{score:.3f}\n")

    return lines

# TODO parametrize number of cores
# TODO try multithreading using joblib
def main(
    fasta_file: Annotated[Path, typer.Option("--input-fasta", "-i")],
    embedding_file: Annotated[Path, typer.Option("--embedding-file", "-e")] = None,
    prostt5_cache_directory: Annotated[Path, typer.Option("--prostt5-cache")] = None,
    model_dir: Annotated[Path, typer.Option("--model-directory")] = None,
    output_dir: Annotated[Path, typer.Option("--output-directory", "-o")] = None,
    write_to_one_file: Annotated[bool, typer.Option("--write-to-one-file")] = False
):
    script_start_time = dt.now()

    model_dir = model_dir or Path.cwd() / "model"
    output_dir = output_dir or Path.cwd() / "out"

    device = "cpu"
    embedding_type = "prostt5"
    method_name = "FNN"
    embedding_dim = embedding_dimensions[embedding_type]

    model_config = yaml.safe_load((model_dir / "config.yml").open())
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        map_location = None
    else:
        device = torch.device("cpu")
        map_location = torch.device("cpu")
        torch.set_default_tensor_type(torch.DoubleTensor)

    model = FNN(n_features=embedding_dim, **model_config["model"]["params"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=map_location))
    model = model.double()

    # Prepare file output
    disorder_output_path = output_dir / "disorder"
    disorder_output_path.mkdir(exist_ok=True, parents=True)

    sequences = OrderedDict((rec.id, str(rec.seq)) for rec in SeqIO.parse(fasta_file, "fasta"))
    execution_times = []

    with progress.Progress(
        *progress.Progress.get_default_columns(), progress.TimeElapsedColumn()
    ) as pbar, torch.no_grad():
        if embedding_file is None:
            embedding_progress = pbar.add_task("Computing embeddings", total=len(sequences))
            pretrained_path = prostt5_cache_directory or "Rostlab/ProstT5"
            tokenizer = T5Tokenizer.from_pretrained(pretrained_path, do_lower_case=False)
            encoder = T5EncoderModel.from_pretrained(pretrained_path)
            encoder.half() if torch.cuda.is_available() else encoder.double()

            embeddings = {}
            with torch.no_grad():
                for id, sequence in sequences.items():
                    seq = "<AA2fold>" + " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
                    ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding="longest", return_tensors='pt').to(device)
                    embeddings[id] = encoder(ids.input_ids, attention_mask=ids.attention_mask).last_hidden_state[0, 1:len(sequence)+1]
                    pbar.advance(embedding_progress)

        else:
            if prostt5_cache_directory:
                logging.warning("Both a cache directory for encoder weights and a file with pre-computed embeddings were provided, so the encoder weights are ignored and the pre-computed embeddings used")

            embeddings = {id: torch.tensor(np.array(emb[()]), device=device) for id, emb in h5py.File(embedding_file).items() if id in sequences.keys()}

        overall_progress = pbar.add_task("Generating predictions", total=len(embeddings))

        all_lines = []

        for id, emb in embeddings.items():
            start_time = dt.now()
            emb = emb.clone().to(device=device, dtype=torch.float64)
            pred = model(emb)

            lines = generate_caid_format(id, pred, sequences[id])
            if write_to_one_file:
                all_lines.extend(lines)
            else:
                with open(disorder_output_path / f"{id}.caid", "w+") as f:
                    f.writelines(lines)
                    
            end_time = dt.now()
            execution_time = end_time - start_time
            execution_times.append({"sequence": id, "milliseconds": execution_time / datetime.timedelta(milliseconds=1)})
            
            pbar.advance(overall_progress)

        csv_lines = pl.from_records(execution_times).write_csv(file=None)

        with open(output_dir / "timings.csv", "w+") as f:
            f.write(f"# Running test, started {script_start_time:%c}\n")
            f.write(csv_lines)

        if write_to_one_file:
            with open(disorder_output_path / "all.caid", "w+") as f:
                f.writelines(all_lines)

if __name__ == "__main__":
    typer.run(main)
