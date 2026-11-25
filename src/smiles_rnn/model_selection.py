import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class RNN(nn.Module):
    def __init__(
        self, input_size: int = 66, hidden_size: int = 128, batch_first: bool = True
    ) -> None:
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.batch_first: bool = batch_first

        self.rnn: nn.RNN = nn.RNN(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
        self.classifier: nn.Linear = nn.Linear(
            in_features=hidden_size, out_features=input_size
        )

        self.hidden: torch.Tensor | None = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output, self.hidden = self.rnn(X)
        out = self.classifier(output)
        logits = out.view(-1, self.input_size)
        return logits


class GRU(nn.Module):
    def __init__(
        self, input_size: int = 66, hidden_size: int = 128, batch_first: bool = True
    ) -> None:
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.batch_first: bool = batch_first

        self.gru: nn.GRU = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
        self.classifier: nn.Linear = nn.Linear(
            in_features=hidden_size, out_features=input_size
        )

        self.hidden: torch.Tensor | None = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output, self.hidden = self.gru(X)
        out = self.classifier(output)
        logits = out.view(-1, self.input_size)
        return logits


class LSTM(nn.Module):
    def __init__(
        self, input_size: int = 66, hidden_size: int = 128, batch_first: bool = True
    ) -> None:
        super().__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.batch_first: bool = batch_first

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
        self.classifier: nn.Linear = nn.Linear(
            in_features=hidden_size, out_features=input_size
        )

        self.hidden: torch.Tensor | None = None
        self.cell: torch.Tensor | None = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output, (self.hidden, self.cell) = self.lstm(X)
        out = self.classifier(output)
        logits = out.view(-1, self.input_size)
        return logits


def model_selection(
    model_type: str = "rnn",
    input_size: int = 66,
    hidden_size: int = 128,
    batch_first: bool = True,
) -> nn.Module:
    if model_type == "rnn":
        model = RNN(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
    elif model_type == "gru":
        model = GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
    elif model_type == "lstm":
        model = LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
    else:
        raise ValueError(
            f"Invalid type of model, we currently not support: {model_type}"
        )
    return model
