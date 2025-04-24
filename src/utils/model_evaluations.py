# File: src/utils/visual_evaluation.py

import csv
import os
from datetime import datetime
from utils.config import Config
import pandas as pd
import matplotlib.pyplot as plt
from utils.config import Config
from utils.logging_config import logger

def save_metrics(run_id, subset_acc, hamming, precision, recall, f1):
    file_path = Config.TRAINING_EVALUATION_DATA
    header = ["Run_ID", "Timestamp", "Subset_Accuracy", "Hamming_Loss", "Precision", "Recall", "F1_Score"]
    
    # Prepare data row
    row = [run_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), subset_acc, hamming, precision, recall, f1]
    
    # If file doesn't exist, write header
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(row)
    else:
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

def get_next_run_id(file_path):
    if not os.path.exists(file_path):
        return "Run_1"
    df = pd.read_csv(file_path)
    last_run = df["Run_ID"].iloc[-1]
    last_num = int(last_run.split("_")[1])
    return f"Run_{last_num + 1}"

def plot_training_metrics(file_path):
    # Load metrics data
    df = pd.read_csv(Config.TRAINING_EVALUATION_DATA)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["Run_ID"], df["Subset_Accuracy"], marker='o', label='Subset Accuracy')
    plt.plot(df["Run_ID"], df["Precision"], marker='o', label='Precision')
    plt.plot(df["Run_ID"], df["Recall"], marker='o', label='Recall')
    plt.plot(df["Run_ID"], df["F1_Score"], marker='o', label='F1 Score')
    plt.plot(df["Run_ID"], df["Hamming_Loss"], marker='o', label='Hamming Loss')

    plt.xlabel("Training Runs")
    plt.xticks(rotation=45)
    plt.ylabel("Metric Value")
    plt.title("Model Metrics Over Training Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_images/train/training_plot.png")

    
    
def training_metric_summary(file_path):
        # Load your metrics data
    df = pd.read_csv(file_path)

    # Calculate mean and standard deviation for each metric
    metrics = ["Subset_Accuracy", "Hamming_Loss", "Precision", "Recall", "F1_Score"]
    summary = df[metrics].agg(['mean', 'std'])

    logger.info("\nMetric Summary Across Runs:")
    logger.info(f"\n{summary}")

def dual_axis_hamming_plot(filepath):
    # Load data
    df = pd.read_csv(filepath)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Accuracy, Precision, Recall, F1
    ax1.plot(df["Run_ID"], df["Subset_Accuracy"], label="Subset Accuracy", marker='o')
    ax1.plot(df["Run_ID"], df["Precision"], label="Precision", marker='o')
    ax1.plot(df["Run_ID"], df["Recall"], label="Recall", marker='o')
    ax1.plot(df["Run_ID"], df["F1_Score"], label="F1 Score", marker='o')
    ax1.set_ylabel("Accuracy / Precision / Recall / F1")
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='x', rotation=45)

    ax1.grid(True)

    # Secondary axis for Hamming Loss
    ax2 = ax1.twinx()
    ax2.plot(df["Run_ID"], df["Hamming_Loss"], label="Hamming Loss", marker='x', color='purple')
    ax2.set_ylabel("Hamming Loss")
    ax2.legend(loc='upper right')

    plt.title("Model Metrics Over Training Runs (Dual Axis)")
    plt.xlabel("Training Runs")
    plt.xticks(rotation=45)
    plt.savefig("output_images/train/dual_axis_hamming_plot.png")

