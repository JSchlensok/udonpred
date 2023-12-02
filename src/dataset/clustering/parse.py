from pathlib import Path
from typing import Union

import polars as pl


def read_cluster_assignments(filepath: Union[Path, str]) -> pl.DataFrame:
    return pl.read_csv(
        filepath,
        separator="\t",
        has_header=False,
        new_columns=["cluster_representative_id", "sequence_id"],
    )
