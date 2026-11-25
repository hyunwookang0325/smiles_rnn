import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = ROOT / "database"
ED_PATH: Path = DATA_ROOT / "word_dict.json"
DD_PATH: Path = DATA_ROOT / "num_dict.json"


def smi_tokenizer(smi: str) -> str:
    """
    Tokenize a SMILES molecule or reaction

    Args:
        smi (str): input smiles to tokenize

    Returns:
        str: seperated tokens
    """
    pattern: str = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex: re.Pattern = re.compile(pattern)
    tokens: list[str] = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens), f"{tokens}, {''.join(tokens)}, {smi}"
    # return " ".join(tokens)
    return tokens


def parsing_json(input_file: Path) -> dict[str, int]:
    with open(input_file, "r") as f:
        data: dict[str, int] = json.load(f)
    return data


if __name__ == "__main__":
    data: dict[str, int] = parsing_json(DICT_PATH)
    logger.info(f"dictionary from database {data}")
    logger.info(f"length of dictionary: {len(list(data.keys()))}")
