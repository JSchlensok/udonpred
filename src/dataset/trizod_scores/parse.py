import json
from pathlib import Path
from typing import Union

import polars as pl


def read_reference_json(filepath: Union[Path, str]) -> pl.DataFrame:
    lines = open(filepath).readlines()
    raw_json = [*[line + ',' for line in lines[:-1]], lines[-1]]
    raw_json.insert(0, '[')
    raw_json.append(']')
    raw_json = ''.join(raw_json)

    df = pl.from_records(json.loads(''.join(raw_json)))
    return df.select(df.columns[:23])  # return just the columns we don't have in the CSV file

def read_score_csv(filepath: Union[Path, str]) -> pl.DataFrame:
    return pl.read_csv(filepath, dtypes={"HA": pl.Float64, "HB": pl.Float64})
