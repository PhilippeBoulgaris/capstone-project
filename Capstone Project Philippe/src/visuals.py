# src/visuals.py
# -*- coding: utf-8 -*-

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

from .performance import compute_drawdown
from .utils import short_model_name


class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_equity_curves(
        self,
        curves_dict: Dict[str, pd.Series],
        filename: str,
        title: str,
        figsize: Tuple[int, int] = (11, 6),
    ):
        plt.figure(figsize=figsize)
        for name, eq in curves_dict.items():
            if eq is not None and len(eq) >= 2:
                (eq / eq.iloc[0]).plot(label=name, linewidth=2.0)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.ylabel("Cumulative Value ($1)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()

    def plot_drawdown_curves(
        self,
        curves_dict: Dict[str, pd.Series],
        filename: str,
        title: str,
        figsize: Tuple[int, int] = (11, 5),
    ):
        plt.figure(figsize=figsize)
        any_plotted = False
        for name, eq in curves_dict.items():
            if eq is None or len(eq) < 2:
                continue
            dd = compute_drawdown(eq)
            if len(dd) < 2:
                continue
            dd.plot(label=name, linewidth=2.0)
            any_plotted = True

        if not any_plotted:
            plt.close()
            return

        plt.title(title, fontsize=13, fontweight="bold")
        plt.ylabel("Drawdown")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()


class MLVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_ml_comparison(self, yahoo_results: Dict, sec_results: Dict, filename: str):
        model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Network"]
        yahoo_acc = [yahoo_results.get(f"{m} (Yahoo-trained)", {}).get("accuracy", np.nan) for m in model_names]
        sec_acc = [sec_results.get(f"{m} (SEC-trained)", {}).get("accuracy", np.nan) for m in model_names]
        yahoo_auc = [yahoo_results.get(f"{m} (Yahoo-trained)", {}).get("auc", np.nan) for m in model_names]
        sec_auc = [sec_results.get(f"{m} (SEC-trained)", {}).get("auc", np.nan) for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax1 = axes[0]
        ax1.bar(x - width / 2, yahoo_acc, width, label="Yahoo")
        ax1.bar(x + width / 2, sec_acc, width, label="SEC")
        ax1.set_title("Accuracy: Yahoo vs SEC")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["LogReg", "RF", "GB", "NN"])
        ax1.axhline(0.5, linestyle="--")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2 = axes[1]
        ax2.bar(x - width / 2, yahoo_auc, width, label="Yahoo")
        ax2.bar(x + width / 2, sec_auc, width, label="SEC")
        ax2.set_title("AUC: Yahoo vs SEC")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["LogReg", "RF", "GB", "NN"])
        ax2.axhline(0.5, linestyle="--")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ ML comparison plot saved: {filename}")

    def plot_ml_metrics_single(self, results: Dict, universe_label: str, filename: str):
        model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Network"]

        acc, auc, short = [], [], []
        for m in model_names:
            key = f"{m} ({universe_label}-trained)"
            acc.append(results.get(key, {}).get("accuracy", np.nan))
            auc.append(results.get(key, {}).get("auc", np.nan))
            short.append(short_model_name(key))

        x = np.arange(len(short))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax1 = axes[0]
        ax1.bar(x, acc)
        ax1.set_title(f"Accuracy — {universe_label} (models compared)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short)
        ax1.axhline(0.5, linestyle="--")
        ax1.grid(alpha=0.3)

        ax2 = axes[1]
        ax2.bar(x, auc)
        ax2.set_title(f"AUC — {universe_label} (models compared)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short)
        ax2.axhline(0.5, linestyle="--")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ ML metrics ({universe_label}) saved: {filename}")

    def plot_roc_curves(self, y_true: np.ndarray, results: Dict, universe_label: str, filename: str):
        plt.figure(figsize=(7, 6))
        model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Network"]

        plotted = False
        for m in model_names:
            key = f"{m} ({universe_label}-trained)"
            if key not in results:
                continue
            proba = results[key].get("probabilities", None)
            if proba is None or len(proba) != len(y_true):
                continue

            fpr, tpr, _ = roc_curve(y_true, proba)
            auc_val = results[key].get("auc", np.nan)
            plt.plot(fpr, tpr, label=f"{short_model_name(key)} (AUC={auc_val:.3f})")
            plotted = True

        if not plotted:
            plt.close()
            return

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"ROC Curves — {universe_label}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ ROC plot ({universe_label}) saved: {filename}")

    def plot_confusion(self, y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        disp.plot(ax=ax, values_format="d")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Confusion matrix saved: {filename}")
