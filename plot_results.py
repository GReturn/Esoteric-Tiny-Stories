import csv
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the loss log
log_path = r'd:\_src\ML\is-gpt\checkpoints_0003\loss_log.csv'
output_path = r'd:\_src\ML\is-gpt\training_performance.png'

if not os.path.exists(log_path):
    print(f"Error: Log file not found at {log_path}")
else:
    steps = []
    train_loss = []
    val_loss = []
    gnorm = []

    with open(log_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['step']:
                steps.append(int(row['step']))
                train_loss.append(float(row['train_loss']))
                val_loss.append(float(row['val_loss']))
                gnorm.append(float(row['gnorm']))

    # Set a clean style
    plt.style.use('bmh') # 'bmh' is built-in and clean
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Learning Curves
    ax1.plot(steps, train_loss, label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(steps, val_loss, label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    ax1.set_title('Cross-Entropy Convergence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient Stability
    ax2.plot(steps, gnorm, color='#2ca02c', label='Gradient Norm', linewidth=1.5)
    ax2.fill_between(steps, gnorm, alpha=0.1, color='#2ca02c')
    ax2.set_title('Training Stability (gnorm)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Norm Value', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Success! Presentation plot saved as: {output_path}")
