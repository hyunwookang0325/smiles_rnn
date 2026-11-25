import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "database" / "zinc250k.csv"
DATA_ROOT: Path = ROOT / "database"
DICT_PATH: Path = DATA_ROOT / "word_dict.json"


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


def prepare_dictionary():
    # ROOT: Path = Path(__file__).resolve().parents[2]
    # DATA: Path = ROOT / "database" / "zinc250k.csv"
    # DATA_ROOT: Path = ROOT / "database"
    # DICT_PATH: Path = DATA_ROOT / "word_dict.json"

    smiles: str = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin smiles
    tokenized_smiles: str = smi_tokenizer(smiles)
    logger.info(f"Current root: {ROOT}")
    df: pd.DataFrame = pd.read_csv(DATA)
    logger.info(f"lenght of the database: {len(df)}")

    # get data from dataframe
    count = 10
    for i, row in df.iterrows():
        if i == count:
            break
        smi = str(row["smiles"].strip())
        logp = row["logP"]
        qed = row["qed"]
        sas = row["SAS"]
        logger.info(f"smiles: {smi} logP: {logp} Q.E.D: {qed} SAS: {sas}")
        logger.info(f"tokenized smiles: {smi_tokenizer(smi)}")
    smiles_list: list[str] = df["smiles"].str.strip().tolist()
    logger.info(f"The length of the list: {len(smiles_list)}")
    dict_set: set[str] = set()

    logger.info(smiles_list[0])
    for smi in smiles_list:
        tokens: set[str] = set(smi_tokenizer(smi))
        dict_set = dict_set | tokens

    word_dict: [str, int] = {
        "PAD_ID": 0,
        "BOS_ID": 1,
        "EOS_ID": 2,
        "IGNORE_ID": -1,
    }
    logger.info(f"Total length of the dictionary: {len(dict_set)}")
    logger.info(f"Dictionary keys: {dict_set}")
    dict_keys: list[str] = list(dict_set)
    dict_keys.sort()

    cnt: int = 3
    for word in dict_keys:
        word_dict[word] = cnt
        cnt += 1

    with open(str(DICT_PATH), "w") as f:
        json.dump(word_dict, f, indent=4)


def prepare_decoder():
    data: dict[str, int] = parsing_json(DICT_PATH)
    logger.info(f"dictionary from database: {data}")
    decode_dict: dict[int, str] = {v: k for k, v in data.items()}
    decode_dict_path: Path = DATA_ROOT / "num_dict.json"
    with open(str(decode_dict_path), "w") as f:
        json.dump(decode_dict, f, indent=4)


if __name__ == "__main__":
    prepare_dictionary()
    prepare_decoder()
