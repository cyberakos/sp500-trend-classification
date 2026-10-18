import json
from pathlib import Path
import numpy as np
import xgboost as xgb

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "reports" / "best_hyperparameters.json"


class XGBoostModel:
    # háromosztályos xgboost modell a triple barrier címkékhez

    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.8,
                 colsample_bytree=0.8, gamma=1.0, reg_lambda=2.0, random_state=42):
        self.params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "gamma": float(gamma),
            "reg_lambda": float(reg_lambda),
            "random_state": int(random_state),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "n_jobs": -1,
        }
        self.model = None
        self.classes_ = np.array([-1, 0, 1])

    @classmethod
    def from_best_params(cls, config_path=None, **overrides):
        # példány létrehozása az optuna által mentett json-ból
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Nem található a hiperparaméter fájl: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = data.get("xgboost", {}).get("params", {})
        params.update(overrides)
        return cls(**params)

    @staticmethod
    def _encode_labels(y):
        # az xgboost-nak 0, 1, 2 címke kell a -1, 0, 1 helyett
        return (y + 1).astype(int)

    @staticmethod
    def _decode_labels(y_encoded):
        return (y_encoded - 1).astype(int)

    def fit(self, X_train, y_train, eval_set=None):
        y_encoded = self._encode_labels(y_train)
        self.model = xgb.XGBClassifier(**self.params)

        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_encoded = self._encode_labels(y_val)
            self.model.fit(
                X_train,
                y_encoded,
                eval_set=[(X_train, y_encoded), (X_val, y_val_encoded)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_encoded, verbose=False)

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("A modell még nincs betanítva.")
        return self.model.predict_proba(X)

    def predict(self, X, tau=None, delta=None):
        probs = self.predict_proba(X)
        if tau is not None and delta is not None:
            p_sell = probs[:, 0]
            p_buy = probs[:, 2]
            preds = np.zeros(len(probs), dtype=int)
            preds[(p_buy >= tau) & ((p_buy - p_sell) > delta)] = 1
            preds[(p_sell >= tau) & ((p_sell - p_buy) > delta)] = -1
            return preds

        encoded_preds = np.argmax(probs, axis=1)
        return self._decode_labels(encoded_preds)

    def get_feature_importance(self, feature_names):
        if self.model is None:
            raise ValueError("A modell még nincs betanítva.")
        return dict(zip(feature_names, self.model.feature_importances_))

    def save(self, path):
        if self.model is None:
            raise ValueError("A modell még nincs betanítva.")
        self.model.save_model(str(path))

    def load(self, path):
        self.model = xgb.XGBClassifier(**self.params)
        self.model.load_model(str(path))
        return self
