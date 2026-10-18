import copy
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "reports" / "best_hyperparameters.json"


class _LSTMNetwork(nn.Module):
    # maga a neurális háló

    def __init__(self, input_dim=20, hidden_dim=64, num_layers=2, dropout=0.25, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # csak az utolsó időlépés kimenetét használjuk
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        return self.fc(out)


class LSTMModel:
    # lstm wrapper, kezeli a csúszóablakokat és a tanítást

    def __init__(self, sequence_length=24, input_dim=20, hidden_dim=64, num_layers=2,
                 dropout=0.25, learning_rate=0.001, batch_size=128, epochs=50,
                 patience=10, random_state=42):
        self.sequence_length = int(sequence_length)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.history_ = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        torch.manual_seed(self.random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None

    @classmethod
    def from_best_params(cls, config_path=None, **overrides):
        # példány létrehozása az optuna által mentett json-ból
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Nem található a hiperparaméter fájl: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = data.get("lstm", {}).get("params", {})
        params.update(overrides)
        return cls(**params)

    @staticmethod
    def _encode_labels(y):
        return (y + 1).astype(int)

    @staticmethod
    def _decode_labels(y_encoded):
        return (y_encoded - 1).astype(int)

    def create_sequences(self, X, y=None):
        # háromdimenziós csúszóablakokat csinál a kétdimenziós adatból
        num_samples = len(X) - self.sequence_length + 1
        if num_samples <= 0:
            raise ValueError("A bemenet hossza rövidebb a szekvencia hosszánál.")

        shape = (num_samples, self.sequence_length, X.shape[1])
        strides = (X.strides[0], X.strides[0], X.strides[1])
        X_seq = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

        if y is not None:
            y_seq = y[self.sequence_length - 1:]
            return X_seq, y_seq
        return X_seq, None

    def fit(self, X_train, y_train, eval_set=None):
        X_tr_seq, y_tr_seq = self.create_sequences(X_train, y_train)
        y_tr_enc = self._encode_labels(y_tr_seq)

        train_dataset = TensorDataset(
            torch.tensor(X_tr_seq, dtype=torch.float32),
            torch.tensor(y_tr_enc, dtype=torch.long),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        has_val = eval_set is not None
        if has_val:
            X_val, y_val = eval_set
            X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
            y_val_enc = self._encode_labels(y_val_seq)
            val_dataset = TensorDataset(
                torch.tensor(X_val_seq, dtype=torch.float32),
                torch.tensor(y_val_enc, dtype=torch.long),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.network = _LSTMNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        best_loss = float("inf")
        best_weights = None
        patience_counter = 0

        self.history_ = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(self.epochs):
            # tanítás
            self.network.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.network(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_train += (preds == batch_y).sum().item()
                total_train += batch_y.size(0)

            epoch_tr_loss = running_loss / total_train
            epoch_tr_acc = correct_train / total_train

            self.history_["train_loss"].append(epoch_tr_loss)
            self.history_["train_acc"].append(epoch_tr_acc)

            # validáció és early stopping
            if has_val:
                self.network.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = self.network(batch_x)
                        val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
                        preds = torch.argmax(outputs, dim=1)
                        correct_val += (preds == batch_y).sum().item()
                        total_val += batch_y.size(0)

                epoch_val_loss = val_loss / total_val
                epoch_val_acc = correct_val / total_val

                self.history_["val_loss"].append(epoch_val_loss)
                self.history_["val_acc"].append(epoch_val_acc)

                if epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    best_weights = copy.deepcopy(self.network.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        # a legjobb validációs súlyok visszatöltése
        if best_weights is not None:
            self.network.load_state_dict(best_weights)

        return self

    def predict_proba(self, X_seq):
        if self.network is None:
            raise ValueError("A modell még nincs betanítva.")

        self.network.eval()
        tensor_x = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.network(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, X_seq, tau=None, delta=None):
        probs = self.predict_proba(X_seq)
        if tau is not None and delta is not None:
            p_sell = probs[:, 0]
            p_buy = probs[:, 2]
            preds = np.zeros(len(probs), dtype=int)
            preds[(p_buy >= tau) & ((p_buy - p_sell) > delta)] = 1
            preds[(p_sell >= tau) & ((p_sell - p_buy) > delta)] = -1
            return preds

        encoded_preds = np.argmax(probs, axis=1)
        return self._decode_labels(encoded_preds)

    def save(self, path):
        if self.network is None:
            raise ValueError("A modell még nincs betanítva.")
        torch.save(self.network.state_dict(), str(path))

    def load(self, path):
        self.network = _LSTMNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.network.load_state_dict(torch.load(str(path), map_location=self.device))
        self.network.eval()
        return self
