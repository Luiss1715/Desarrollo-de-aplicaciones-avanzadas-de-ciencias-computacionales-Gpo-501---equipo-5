from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class LatentNNConfig:
    input_dim: int
    embedding_model: str
    hidden_dim_1: int = 256
    hidden_dim_2: int = 64
    dropout: float = 0.3
    threshold: float = 0.5


class LatentMLP(nn.Module):
    def __init__(self, config: LatentNNConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim_1, config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim_2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def train_mlp(
    train_embeddings: np.ndarray,
    train_labels: list[int],
    validation_embeddings: np.ndarray,
    validation_labels: list[int],
    config: LatentNNConfig,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> tuple[LatentMLP, list[dict[str, float]]]:
    torch.manual_seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LatentMLP(config).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(
        torch.as_tensor(train_embeddings, dtype=torch.float32),
        torch.as_tensor(train_labels, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = []

    for epoch in range(epochs):
        model.train()
        losses = []
        for features, labels in loader:
            features, labels = features.to(target_device), labels.to(target_device)
            optimizer.zero_grad()
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        scores = predict_scores(model, validation_embeddings, device=str(target_device))
        predictions = (scores >= config.threshold).astype(int)
        accuracy = float(np.mean(predictions == np.asarray(validation_labels)))
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(losses)),
                "validation_accuracy": accuracy,
            }
        )
    return model, history


def predict_scores(
    model: LatentMLP,
    embeddings: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            features = torch.as_tensor(
                embeddings[start : start + batch_size], dtype=torch.float32, device=target_device
            )
            scores.extend(torch.sigmoid(model(features)).cpu().numpy().tolist())
    return np.asarray(scores, dtype=float)


def predict_labels(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (scores >= threshold).astype(int)


def save_latent_model(
    model: LatentMLP,
    config: LatentNNConfig,
    model_path: str | Path,
    config_path: str | Path,
) -> None:
    model_output = Path(model_path)
    config_output = Path(config_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output)
    config_output.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_latent_model(
    model_path: str | Path,
    config_path: str | Path,
    device: str | None = None,
) -> tuple[LatentMLP, LatentNNConfig]:
    config = LatentNNConfig(**json.loads(Path(config_path).read_text(encoding="utf-8")))
    model = LatentMLP(config)
    try:
        state = torch.load(model_path, map_location=device or "cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device or "cpu")
    model.load_state_dict(state)
    model.eval()
    return model, config
