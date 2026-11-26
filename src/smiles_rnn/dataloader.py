from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .encoder import parsing_json, smi_tokenizer

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = ROOT / "database"
DATA: Path = DATA_ROOT / "zinc250k.csv"
ED_PATH: Path = DATA_ROOT / "word_dict.json"


def database_loader(input_file: Path) -> list[str]:
    df: pd.DataFrame = pd.read_csv(input_file)
    logger.info("The database is successfully imported")
    smiles: list[str] = df["smiles"].str.strip().tolist()
    logger.info(f"Sample smiles: {smiles[0]}")

    return smiles


def input_tokenizer(input_list: list[str], word_dict: dict[str, int]) -> np.ndarray:
    max_length: int = max([len(i) for i in input_list]) + 2
    tmp_list: list[np.ndarray] = [
        np.zeros(shape=(max_length, len(word_dict)), dtype=np.float32)
        for _ in range(len(input_list))
    ]
    input_array: np.ndarray = np.array(tmp_list, dtype=np.float32)
    logger.info(f"Shape of the array: {input_array.shape}")
    for i, smi in enumerate(input_list):
        # BOS
        input_array[i][0][1] = 1
        # check for the code: logger.info(input_array[i][0])
        tokens: list[str] = smi_tokenizer(smi)
        for j, token in enumerate(tokens):
            num: int = word_dict[token]
            input_array[i][j + 1][num] = 1
        for j in range(j + 2, max_length):
            num = word_dict["PAD_ID"]
            input_array[i][j][num] = 1
    return input_array


def output_tokenizer(input_list: list[str], word_dict: dict[str, int]) -> np.ndarray:
    max_length: int = max([len(i) for i in input_list]) + 2
    tmp_list: list[np.ndarray] = [
        np.zeros(shape=(max_length,), dtype=np.float32) for _ in range(len(input_list))
    ]
    output_array: np.ndarray = np.array(tmp_list, dtype=np.int64)
    for i, smi in enumerate(input_list):
        tokens: list[str] = smi_tokenizer(smi)
        for j, token in enumerate(tokens):
            num: int = word_dict[token]
            output_array[i][j] = num
        num = word_dict["EOS_ID"]
        output_array[i][j + 1] = num
        for n in range(j + 2, max_length):
            num = word_dict["IGNORE_ID"]
            output_array[i][n] = num

    return output_array


def make_dataset(
    input_array: np.ndarray,
    output_array: np.ndarray,
    batch_size: int = 32,
    ratio: float = 0.3,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    X_train, X_val, y_train, y_val = train_test_split(
        input_array, output_array, test_size=ratio, random_state=seed
    )
    # convert numpy to torch
    X_train: torch.Tensor = torch.as_tensor(X_train, dtype=torch.float32)
    y_train: torch.Tensor = torch.as_tensor(y_train, dtype=torch.int64)
    X_val: torch.Tensor = torch.as_tensor(X_val, dtype=torch.float32)
    y_val: torch.Tensor = torch.as_tensor(y_val, dtype=torch.int64)

    train_data: TensorDataset = TensorDataset(X_train, y_train)
    val_data: TensorDataset = TensorDataset(X_val, y_val)

    train_loader: DataLoader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader: DataLoader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader


if __name__ == "__main__":
    smiles: list[str] = database_loader(input_file=DATA)
    data: dict[str, int] = parsing_json(ED_PATH)
    # logger.info(data)
    input_array = input_tokenizer(input_list=smiles[:10], word_dict=data)
    # logger.info(input_array[0])
    output_array = output_tokenizer(input_list=smiles[:10], word_dict=data)
    # logger.info(output_array[0])
    train_loader, val_loader = make_dataset(input_array, output_array)
    logger.info(f"train: {train_loader} val: {val_loader}")
