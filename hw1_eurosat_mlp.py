#!/usr/bin/env python3
"""
HW1: 从零开始实现三层神经网络(MLP)完成 EuroSAT_RGB 地表覆盖分类

特点:
- 不依赖 PyTorch / TensorFlow / JAX
- 仅使用 NumPy 实现前向传播与反向传播
- 支持 ReLU / Sigmoid / Tanh
- 支持 SGD、学习率衰减、L2 正则
- 支持网格搜索、最优权重保存、测试集评估、混淆矩阵
- 支持训练曲线、第一层权重可视化、错例分析

推荐运行方式:
1) 先做超参数搜索 + 训练 + 测试
python hw1_eurosat_mlp.py \
    --mode search \
    --data_root /path/to/EuroSAT_RGB \
    --output_dir outputs \
    --image_size 32 \
    --epochs 25 \
    --batch_size 128 \
    --search_lrs 0.1,0.05,0.01 \
    --search_hiddens 128,256 \
    --search_wds 0,1e-4,1e-3 \
    --search_activations relu,tanh

2) 单组参数训练
python hw1_eurosat_mlp.py \
    --mode train \
    --data_root /path/to/EuroSAT_RGB \
    --output_dir outputs_single \
    --image_size 32 \
    --hidden_dim 256 \
    --activation relu \
    --lr 0.05 \
    --weight_decay 1e-4 \
    --epochs 25

3) 只加载已保存的最优模型，在测试集上评估
python hw1_eurosat_mlp.py \
    --mode test \
    --data_root /path/to/EuroSAT_RGB \
    --weights /path/to/best_model.npz \
    --output_dir outputs_test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# =========================
# 工具函数
# =========================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_list(value: str, cast_fn):
    if value is None or value == "":
        return []
    return [cast_fn(v.strip()) for v in value.split(",") if v.strip() != ""]


# =========================
# 数据加载与预处理
# =========================

def list_class_folders(data_root: Path) -> List[Path]:
    folders = [p for p in data_root.iterdir() if p.is_dir()]
    folders.sort(key=lambda x: x.name)
    if not folders:
        raise FileNotFoundError(f"在 {data_root} 下没有找到类别文件夹")
    return folders


def load_eurosat_dataset(
    data_root: str,
    image_size: int = 32,
    max_per_class: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    将 EuroSAT_RGB 文件夹读取为:
    X: [N, D] float32
    y: [N] int64
    class_names: 类别名
    paths: 每张图像路径
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"数据集路径不存在: {data_root}")

    class_folders = list_class_folders(data_root)
    class_names = [p.name for p in class_folders]

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    path_list: List[str] = []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    print(f"[Data] 发现 {len(class_names)} 个类别: {class_names}")
    for class_idx, folder in enumerate(class_folders):
        image_files = [p for p in folder.iterdir() if p.suffix.lower() in valid_exts]
        image_files.sort(key=lambda x: x.name)
        if max_per_class is not None:
            image_files = image_files[:max_per_class]

        if not image_files:
            print(f"[Warning] 类别 {folder.name} 下没有图像文件，已跳过")
            continue

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                if image_size is not None and image_size > 0:
                    img = img.resize((image_size, image_size), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                X_list.append(arr.reshape(-1))
                y_list.append(class_idx)
                path_list.append(str(img_path))
            except Exception as exc:
                print(f"[Warning] 读取失败 {img_path}: {exc}")

    if not X_list:
        raise RuntimeError("没有成功读取到任何图像，请检查数据集路径和图像格式")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"[Data] 读取完成: X.shape={X.shape}, y.shape={y.shape}")
    return X, y, class_names, path_list


def stratified_split_indices(
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        if n_train == 0 or n_val == 0 or n_test == 0:
            raise ValueError(
                f"类别 {c} 的样本太少，无法完成 train/val/test 划分: n={n}"
            )

        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def standardize_by_train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)

    X_train_std = ((X_train - train_mean) / train_std).astype(np.float32)
    X_val_std = ((X_val - train_mean) / train_std).astype(np.float32)
    X_test_std = ((X_test - train_mean) / train_std).astype(np.float32)
    return X_train_std, X_val_std, X_test_std, train_mean.astype(np.float32), train_std.astype(np.float32)


# =========================
# 模型定义
# =========================

class Linear:
    def __init__(self, in_dim: int, out_dim: int, activation_hint: str = "relu"):
        if activation_hint.lower() == "relu":
            scale = np.sqrt(2.0 / in_dim)
        else:
            scale = np.sqrt(1.0 / in_dim)

        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros((1, out_dim), dtype=np.float32)

        self.x: np.ndarray | None = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("Linear.backward() 调用前必须先 forward()")
        self.dW = self.x.T @ grad_out
        self.db = np.sum(grad_out, axis=0, keepdims=True)
        grad_x = grad_out @ self.W.T
        return grad_x


class ReLU:
    def __init__(self):
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise RuntimeError("ReLU.backward() 调用前必须先 forward()")
        return grad_out * self.mask


class Sigmoid:
    def __init__(self):
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_clip = np.clip(x, -40, 40)
        self.out = 1.0 / (1.0 + np.exp(-x_clip))
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("Sigmoid.backward() 调用前必须先 forward()")
        return grad_out * self.out * (1.0 - self.out)


class Tanh:
    def __init__(self):
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.out is None:
            raise RuntimeError("Tanh.backward() 调用前必须先 forward()")
        return grad_out * (1.0 - self.out ** 2)


class MLP:
    """
    三层神经网络(按常见课程口径):
    输入层 -> 隐藏层 -> 输出层
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, activation: str = "relu"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.activation_name = activation.lower()

        self.fc1 = Linear(input_dim, hidden_dim, activation_hint=self.activation_name)
        self.act = self._build_activation(self.activation_name)
        self.fc2 = Linear(hidden_dim, num_classes, activation_hint="linear")

    @staticmethod
    def _build_activation(name: str):
        name = name.lower()
        if name == "relu":
            return ReLU()
        if name == "sigmoid":
            return Sigmoid()
        if name == "tanh":
            return Tanh()
        raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = self.fc1.forward(x)
        a1 = self.act.forward(z1)
        logits = self.fc2.forward(a1)
        return logits

    def backward(self, grad_logits: np.ndarray) -> None:
        grad_a1 = self.fc2.backward(grad_logits)
        grad_z1 = self.act.backward(grad_a1)
        _ = self.fc1.backward(grad_z1)

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        preds = []
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size]
            logits = self.forward(xb)
            pred = np.argmax(logits, axis=1)
            preds.append(pred)
        return np.concatenate(preds, axis=0)

    def apply_gradients(self, lr: float, weight_decay: float) -> None:
        self.fc1.W -= lr * (self.fc1.dW + weight_decay * self.fc1.W)
        self.fc1.b -= lr * self.fc1.db
        self.fc2.W -= lr * (self.fc2.dW + weight_decay * self.fc2.W)
        self.fc2.b -= lr * self.fc2.db

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "W1": self.fc1.W.copy(),
            "b1": self.fc1.b.copy(),
            "W2": self.fc2.W.copy(),
            "b2": self.fc2.b.copy(),
        }

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.fc1.W = state["W1"].astype(np.float32)
        self.fc1.b = state["b1"].astype(np.float32)
        self.fc2.W = state["W2"].astype(np.float32)
        self.fc2.b = state["b2"].astype(np.float32)


# =========================
# 损失函数与评估
# =========================

def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    n = logits.shape[0]
    eps = 1e-12
    correct_log_probs = -np.log(probs[np.arange(n), y] + eps)
    loss = float(np.mean(correct_log_probs))

    grad_logits = probs.copy()
    grad_logits[np.arange(n), y] -= 1.0
    grad_logits /= n
    return loss, grad_logits.astype(np.float32), probs.astype(np.float32)


def compute_l2_loss(model: MLP, weight_decay: float) -> float:
    if weight_decay <= 0:
        return 0.0
    return 0.5 * weight_decay * float(np.sum(model.fc1.W ** 2) + np.sum(model.fc2.W ** 2))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1)
    denom = np.where(denom == 0, 1, denom)
    return np.diag(cm) / denom


# =========================
# 训练与验证
# =========================

def make_batches(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int | None = None):
    n = len(X)
    indices = np.arange(n)
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


@np.errstate(over='ignore', invalid='ignore')
def evaluate(model: MLP, X: np.ndarray, y: np.ndarray, weight_decay: float, batch_size: int = 512):
    losses = []
    preds = []
    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model.forward(xb)
        loss_ce, _, _ = softmax_cross_entropy(logits, yb)
        loss = loss_ce + compute_l2_loss(model, weight_decay)
        losses.append(loss)
        preds.append(np.argmax(logits, axis=1))

    y_pred = np.concatenate(preds, axis=0)
    avg_loss = float(np.mean(losses))
    acc = accuracy_score(y, y_pred)
    return avg_loss, acc, y_pred


def train_one_model(
    model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_decay: float,
    weight_decay: float,
    save_best_path: Path | None = None,
    save_meta: Dict | None = None,
) -> Tuple[Dict[str, List[float]], float, Dict[str, np.ndarray]]:
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_state = model.state_dict()

    current_lr = lr
    for epoch in range(1, epochs + 1):
        batch_losses = []
        batch_preds = []
        batch_true = []

        for xb, yb in make_batches(X_train, y_train, batch_size=batch_size):
            logits = model.forward(xb)
            loss_ce, grad_logits, _ = softmax_cross_entropy(logits, yb)
            l2_loss = compute_l2_loss(model, weight_decay)
            loss = loss_ce + l2_loss

            model.backward(grad_logits)
            model.apply_gradients(lr=current_lr, weight_decay=weight_decay)

            batch_losses.append(loss)
            batch_preds.append(np.argmax(logits, axis=1))
            batch_true.append(yb)

        train_pred = np.concatenate(batch_preds, axis=0)
        train_true = np.concatenate(batch_true, axis=0)
        train_acc = accuracy_score(train_true, train_pred)
        train_loss = float(np.mean(batch_losses))

        val_loss, val_acc, _ = evaluate(model, X_val, y_val, weight_decay=weight_decay, batch_size=batch_size)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            if save_best_path is not None and save_meta is not None:
                save_model(save_best_path, model, save_meta)
                print(f"  -> 保存新的最优模型到: {save_best_path}")

        current_lr *= lr_decay

    model.load_state_dict(best_state)
    return history, best_val_acc, best_state


# =========================
# 模型保存 / 加载
# =========================

def save_model(save_path: Path, model: MLP, meta: Dict) -> None:
    state = model.state_dict()
    np.savez_compressed(
        save_path,
        W1=state["W1"],
        b1=state["b1"],
        W2=state["W2"],
        b2=state["b2"],
        hidden_dim=np.array([meta["hidden_dim"]], dtype=np.int64),
        input_dim=np.array([meta["input_dim"]], dtype=np.int64),
        num_classes=np.array([meta["num_classes"]], dtype=np.int64),
        image_size=np.array([meta["image_size"]], dtype=np.int64),
        activation=np.array([meta["activation"]], dtype=object),
        class_names=np.array(meta["class_names"], dtype=object),
        train_mean=meta["train_mean"],
        train_std=meta["train_std"],
        train_idx=np.array(meta["train_idx"], dtype=np.int64),
        val_idx=np.array(meta["val_idx"], dtype=np.int64),
        test_idx=np.array(meta["test_idx"], dtype=np.int64),
        weight_decay=np.array([meta["weight_decay"]], dtype=np.float32),
        seed=np.array([meta["seed"]], dtype=np.int64),
    )


def load_model(weights_path: str | Path) -> Tuple[MLP, Dict]:
    data = np.load(weights_path, allow_pickle=True)
    hidden_dim = int(data["hidden_dim"][0])
    input_dim = int(data["input_dim"][0])
    num_classes = int(data["num_classes"][0])
    image_size = int(data["image_size"][0])
    activation = str(data["activation"][0])

    model = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        activation=activation,
    )
    model.load_state_dict({
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"],
    })

    meta = {
        "hidden_dim": hidden_dim,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "image_size": image_size,
        "activation": activation,
        "class_names": list(data["class_names"]),
        "train_mean": data["train_mean"].astype(np.float32),
        "train_std": data["train_std"].astype(np.float32),
        "train_idx": data["train_idx"].astype(np.int64),
        "val_idx": data["val_idx"].astype(np.int64),
        "test_idx": data["test_idx"].astype(np.int64),
        "weight_decay": float(data["weight_decay"][0]),
        "seed": int(data["seed"][0]),
    }
    return model, meta


# =========================
# 可视化与报告输出
# =========================

def save_history_json(history: Dict[str, List[float]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def plot_training_curves(history: Dict[str, List[float]], save_dir: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "val_accuracy_curve.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], save_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    max_val = cm.max() if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > max_val / 2 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def visualize_first_layer_weights(model: MLP, image_size: int, save_path: Path, max_neurons: int = 16) -> None:
    W1 = model.fc1.W  # [input_dim, hidden_dim]
    hidden_dim = W1.shape[1]
    n_show = min(max_neurons, hidden_dim)
    cols = 4
    rows = int(np.ceil(n_show / cols))

    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(n_show):
        w = W1[:, i].reshape(image_size, image_size, 3)
        w_min, w_max = w.min(), w.max()
        if w_max - w_min < 1e-8:
            w_vis = np.zeros_like(w)
        else:
            w_vis = (w - w_min) / (w_max - w_min)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(w_vis)
        plt.title(f"Neuron {i}")
        plt.axis("off")

    plt.suptitle("First-layer Weight Visualization")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_error_analysis_figure(
    paths: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    save_path: Path,
    image_size: int,
    max_examples: int = 12,
) -> None:
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("[Error Analysis] 测试集中没有错分样本")
        return

    wrong = wrong[:max_examples]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, idx in enumerate(wrong):
        img = Image.open(paths[idx]).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"T: {class_names[y_true[idx]]}\nP: {class_names[y_pred[idx]]}", fontsize=9)
        plt.axis("off")

    plt.suptitle("Error Analysis: Misclassified Test Images")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_metrics_text(
    save_path: Path,
    val_acc: float,
    test_acc: float,
    cm: np.ndarray,
    class_names: Sequence[str],
) -> None:
    per_cls = per_class_accuracy(cm)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Validation Accuracy: {val_acc:.6f}\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nPer-class Accuracy:\n")
        for name, acc in zip(class_names, per_cls):
            f.write(f"{name}: {acc:.6f}\n")


# =========================
# 搜索 / 训练 / 测试
# =========================

def prepare_data_for_training(args):
    X, y, class_names, paths = load_eurosat_dataset(
        data_root=args.data_root,
        image_size=args.image_size,
        max_per_class=args.max_per_class,
    )

    train_idx, val_idx, test_idx = stratified_split_indices(
        y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    X_train, X_val, X_test, train_mean, train_std = standardize_by_train(X_train, X_val, X_test)

    print(f"[Split] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_mean": train_mean,
        "train_std": train_std,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "class_names": class_names,
        "paths": paths,
        "num_classes": len(class_names),
        "input_dim": X_train.shape[1],
    }


def train_and_report(
    output_dir: Path,
    data_bundle: Dict,
    hidden_dim: int,
    activation: str,
    lr: float,
    weight_decay: float,
    args,
    tag: str,
) -> Dict:
    model = MLP(
        input_dim=data_bundle["input_dim"],
        hidden_dim=hidden_dim,
        num_classes=data_bundle["num_classes"],
        activation=activation,
    )

    run_dir = ensure_dir(output_dir / tag)
    best_model_path = run_dir / "best_model.npz"

    save_meta = {
        "hidden_dim": hidden_dim,
        "input_dim": data_bundle["input_dim"],
        "num_classes": data_bundle["num_classes"],
        "image_size": args.image_size,
        "activation": activation,
        "class_names": data_bundle["class_names"],
        "train_mean": data_bundle["train_mean"],
        "train_std": data_bundle["train_std"],
        "train_idx": data_bundle["train_idx"],
        "val_idx": data_bundle["val_idx"],
        "test_idx": data_bundle["test_idx"],
        "weight_decay": weight_decay,
        "seed": args.seed,
    }

    history, best_val_acc, _ = train_one_model(
        model=model,
        X_train=data_bundle["X_train"],
        y_train=data_bundle["y_train"],
        X_val=data_bundle["X_val"],
        y_val=data_bundle["y_val"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=lr,
        lr_decay=args.lr_decay,
        weight_decay=weight_decay,
        save_best_path=best_model_path,
        save_meta=save_meta,
    )

    # 重新加载最佳模型，保证后续分析用的确实是最佳权重
    model, meta = load_model(best_model_path)

    val_loss, val_acc, val_pred = evaluate(
        model,
        data_bundle["X_val"],
        data_bundle["y_val"],
        weight_decay=weight_decay,
        batch_size=args.batch_size,
    )
    test_loss, test_acc, test_pred = evaluate(
        model,
        data_bundle["X_test"],
        data_bundle["y_test"],
        weight_decay=weight_decay,
        batch_size=args.batch_size,
    )

    cm = confusion_matrix_np(data_bundle["y_test"], test_pred, data_bundle["num_classes"])

    save_history_json(history, run_dir / "history.json")
    plot_training_curves(history, run_dir)
    plot_confusion_matrix(cm, data_bundle["class_names"], run_dir / "confusion_matrix.png")
    visualize_first_layer_weights(model, args.image_size, run_dir / "first_layer_weights.png")

    test_paths = [data_bundle["paths"][i] for i in data_bundle["test_idx"]]
    save_error_analysis_figure(
        paths=test_paths,
        y_true=data_bundle["y_test"],
        y_pred=test_pred,
        class_names=data_bundle["class_names"],
        save_path=run_dir / "error_analysis.png",
        image_size=args.image_size,
        max_examples=12,
    )

    save_metrics_text(
        run_dir / "metrics.txt",
        val_acc=val_acc,
        test_acc=test_acc,
        cm=cm,
        class_names=data_bundle["class_names"],
    )

    result = {
        "tag": tag,
        "run_dir": str(run_dir),
        "weights_path": str(best_model_path),
        "hidden_dim": hidden_dim,
        "activation": activation,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": cm,
        "class_names": data_bundle["class_names"],
        "meta": meta,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "history": history,
    }
    return result


def run_search(args) -> None:
    output_dir = ensure_dir(args.output_dir)
    data_bundle = prepare_data_for_training(args)

    lrs = parse_list(args.search_lrs, float) or [args.lr]
    hiddens = parse_list(args.search_hiddens, int) or [args.hidden_dim]
    wds = parse_list(args.search_wds, float) or [args.weight_decay]
    activations = parse_list(args.search_activations, str) or [args.activation]

    print("\n[Search] 网格搜索空间:")
    print(f"  lrs={lrs}")
    print(f"  hiddens={hiddens}")
    print(f"  weight_decays={wds}")
    print(f"  activations={activations}")

    search_csv = output_dir / "search_results.csv"
    with open(search_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tag", "hidden_dim", "activation", "lr", "weight_decay",
            "val_loss", "val_acc", "test_loss", "test_acc", "weights_path"
        ])

        best_result = None
        run_id = 0
        for hidden_dim in hiddens:
            for activation in activations:
                for lr in lrs:
                    for wd in wds:
                        run_id += 1
                        tag = f"run_{run_id:03d}_hd{hidden_dim}_act{activation}_lr{lr}_wd{wd}"
                        print("\n" + "=" * 90)
                        print(f"[Search] 开始训练 {tag}")
                        print("=" * 90)
                        result = train_and_report(
                            output_dir=output_dir,
                            data_bundle=data_bundle,
                            hidden_dim=hidden_dim,
                            activation=activation,
                            lr=lr,
                            weight_decay=wd,
                            args=args,
                            tag=tag,
                        )

                        writer.writerow([
                            result["tag"], result["hidden_dim"], result["activation"], result["lr"], result["weight_decay"],
                            result["val_loss"], result["val_acc"], result["test_loss"], result["test_acc"], result["weights_path"]
                        ])
                        f.flush()

                        if best_result is None or result["val_acc"] > best_result["val_acc"]:
                            best_result = result

    if best_result is None:
        raise RuntimeError("搜索过程中没有产生任何结果")

    summary = {
        "best_tag": best_result["tag"],
        "hidden_dim": best_result["hidden_dim"],
        "activation": best_result["activation"],
        "lr": best_result["lr"],
        "weight_decay": best_result["weight_decay"],
        "val_acc": best_result["val_acc"],
        "test_acc": best_result["test_acc"],
        "weights_path": best_result["weights_path"],
        "run_dir": best_result["run_dir"],
    }
    with open(output_dir / "best_result.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Search] 搜索完成，最佳结果如下:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Search] 搜索结果表: {search_csv}")



def run_train(args) -> None:
    output_dir = ensure_dir(args.output_dir)
    data_bundle = prepare_data_for_training(args)
    tag = f"single_hd{args.hidden_dim}_act{args.activation}_lr{args.lr}_wd{args.weight_decay}"

    result = train_and_report(
        output_dir=output_dir,
        data_bundle=data_bundle,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        lr=args.lr,
        weight_decay=args.weight_decay,
        args=args,
        tag=tag,
    )

    print("\n[Train] 训练完成")
    print(f"[Train] 最佳权重: {result['weights_path']}")
    print(f"[Train] 验证集准确率: {result['val_acc']:.4f}")
    print(f"[Train] 测试集准确率: {result['test_acc']:.4f}")
    print("[Train] 测试集混淆矩阵:")
    print(result["confusion_matrix"])



def run_test(args) -> None:
    if args.weights is None:
        raise ValueError("--mode test 时必须提供 --weights")

    output_dir = ensure_dir(args.output_dir)
    weights_path = Path(args.weights)
    model, meta = load_model(weights_path)

    # 使用权重文件中保存的 image_size / split / 标准化参数
    X, y, class_names, paths = load_eurosat_dataset(
        data_root=args.data_root,
        image_size=meta["image_size"],
        max_per_class=args.max_per_class,
    )

    train_idx = meta["train_idx"]
    val_idx = meta["val_idx"]
    test_idx = meta["test_idx"]

    X_test = X[test_idx]
    y_test = y[test_idx]

    X_test = ((X_test - meta["train_mean"]) / meta["train_std"]).astype(np.float32)
    test_loss, test_acc, test_pred = evaluate(
        model,
        X_test,
        y_test,
        weight_decay=meta["weight_decay"],
        batch_size=args.batch_size,
    )
    cm = confusion_matrix_np(y_test, test_pred, len(class_names))

    plot_confusion_matrix(cm, class_names, output_dir / "test_confusion_matrix.png")
    test_paths = [paths[i] for i in test_idx]
    save_error_analysis_figure(
        paths=test_paths,
        y_true=y_test,
        y_pred=test_pred,
        class_names=class_names,
        save_path=output_dir / "test_error_analysis.png",
        image_size=meta["image_size"],
        max_examples=12,
    )
    save_metrics_text(output_dir / "test_metrics.txt", val_acc=np.nan, test_acc=test_acc, cm=cm, class_names=class_names)

    print("[Test] 测试完成")
    print(f"[Test] 权重文件: {weights_path}")
    print(f"[Test] 测试集 loss: {test_loss:.4f}")
    print(f"[Test] 测试集 accuracy: {test_acc:.4f}")
    print("[Test] 混淆矩阵:")
    print(cm)


# =========================
# 命令行参数
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EuroSAT_RGB MLP from scratch")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "search", "test"])
    parser.add_argument("--data_root", type=str, required=True, help="EuroSAT_RGB 文件夹路径")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--weights", type=str, default=None, help="测试模式加载的权重路径")

    parser.add_argument("--image_size", type=int, default=32, help="建议 32 或 48；MLP 不建议太大")
    parser.add_argument("--max_per_class", type=int, default=None, help="调试时可限制每类样本数")

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--search_lrs", type=str, default="0.1,0.05,0.01")
    parser.add_argument("--search_hiddens", type=str, default="128,256")
    parser.add_argument("--search_wds", type=str, default="0,1e-4,1e-3")
    parser.add_argument("--search_activations", type=str, default="relu,tanh")
    return parser


# =========================
# main
# =========================

def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    if args.mode == "search":
        run_search(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    else:
        raise ValueError(f"未知 mode: {args.mode}")


if __name__ == "__main__":
    main()
