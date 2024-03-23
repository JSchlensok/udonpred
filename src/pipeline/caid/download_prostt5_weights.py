from pathlib import Path
from typing_extensions import Annotated

import typer
from transformers import T5Tokenizer, T5EncoderModel

def main(
    cache_dir: Annotated[Path, typer.Argument()] = "prostt5_cache"
):
    cache_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
    encoder = T5EncoderModel.from_pretrained("Rostlab/ProstT5")

    tokenizer.save_pretrained(cache_dir)
    encoder.save_pretrained(cache_dir)

if __name__ == "__main__":
    typer.run(main)
