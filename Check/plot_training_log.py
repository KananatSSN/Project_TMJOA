import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def visualize_training_metrics(csv_file_path):
    """
    Visualize training metrics from a CSV file containing epoch, accuracy, loss, val_accuracy, val_loss
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Clean column names (remove any extra whitespace)
    df.columns = df.columns.str.strip()
    
    # Print basic info about the data
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Visualization', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy over epochs
    axes[0, 0].plot(df['epoch'], df['accuracy'], label='Training Accuracy', marker='o', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)
    axes[0, 0].set_title('Accuracy Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss over epochs
    axes[0, 1].plot(df['epoch'], df['loss'], label='Training Loss', marker='o', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    axes[0, 1].set_title('Loss Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Combined accuracy comparison
    axes[1, 0].plot(df['epoch'], df['accuracy'], label='Training', marker='o', linewidth=2)
    axes[1, 0].plot(df['epoch'], df['val_accuracy'], label='Validation', marker='s', linewidth=2)
    axes[1, 0].fill_between(df['epoch'], df['accuracy'], alpha=0.3)
    axes[1, 0].fill_between(df['epoch'], df['val_accuracy'], alpha=0.3)
    axes[1, 0].set_title('Training vs Validation Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Combined loss comparison
    axes[1, 1].plot(df['epoch'], df['loss'], label='Training', marker='o', linewidth=2)
    axes[1, 1].plot(df['epoch'], df['val_loss'], label='Validation', marker='s', linewidth=2)
    axes[1, 1].fill_between(df['epoch'], df['loss'], alpha=0.3)
    axes[1, 1].fill_between(df['epoch'], df['val_loss'], alpha=0.3)
    axes[1, 1].set_title('Training vs Validation Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    print("\nTraining Summary:")
    print(f"Best Training Accuracy: {df['accuracy'].max():.4f} at epoch {df.loc[df['accuracy'].idxmax(), 'epoch']}")
    print(f"Best Validation Accuracy: {df['val_accuracy'].max():.4f} at epoch {df.loc[df['val_accuracy'].idxmax(), 'epoch']}")
    print(f"Lowest Training Loss: {df['loss'].min():.4f} at epoch {df.loc[df['loss'].idxmin(), 'epoch']}")
    print(f"Lowest Validation Loss: {df['val_loss'].min():.4f} at epoch {df.loc[df['val_loss'].idxmin(), 'epoch']}")
    
    # Calculate overfitting indicators
    final_epoch = df['epoch'].max()
    final_data = df[df['epoch'] == final_epoch].iloc[0]
    
    acc_gap = final_data['accuracy'] - final_data['val_accuracy']
    loss_gap = final_data['val_loss'] - final_data['loss']
    
    print(f"\nFinal Epoch Analysis:")
    print(f"Training Accuracy: {final_data['accuracy']:.4f}")
    print(f"Validation Accuracy: {final_data['val_accuracy']:.4f}")
    print(f"Accuracy Gap: {acc_gap:.4f}")
    print(f"Training Loss: {final_data['loss']:.4f}")
    print(f"Validation Loss: {final_data['val_loss']:.4f}")
    print(f"Loss Gap: {loss_gap:.4f}")
    
    if acc_gap > 0.05 or loss_gap > 0.1:
        print("\n⚠️  Potential overfitting detected!")
    else:
        print("\n✅ Model appears to be training well!")

def create_advanced_plots(csv_file_path):
    """
    Create more advanced visualizations for deeper analysis
    """
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()
    
    # Create a more detailed analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Advanced Training Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Learning curves with confidence intervals
    axes[0, 0].plot(df['epoch'], df['accuracy'], label='Training', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_accuracy'], label='Validation', marker='s')
    axes[0, 0].set_title('Learning Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    axes[0, 1].plot(df['epoch'], df['loss'], label='Training', marker='o')
    axes[0, 1].plot(df['epoch'], df['val_loss'], label='Validation', marker='s')
    axes[0, 1].set_title('Loss Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Accuracy difference (overfitting indicator)
    acc_diff = df['accuracy'] - df['val_accuracy']
    axes[0, 2].plot(df['epoch'], acc_diff, marker='o', color='red', linewidth=2)
    axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 2].set_title('Accuracy Gap (Training - Validation)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy Difference')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Loss difference
    loss_diff = df['val_loss'] - df['loss']
    axes[1, 0].plot(df['epoch'], loss_diff, marker='o', color='orange', linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Loss Gap (Validation - Training)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Training progress (normalized)
    axes[1, 1].plot(df['epoch'], (df['accuracy'] - df['accuracy'].min()) / (df['accuracy'].max() - df['accuracy'].min()), 
                   label='Training Acc (normalized)', marker='o')
    axes[1, 1].plot(df['epoch'], (df['val_accuracy'] - df['val_accuracy'].min()) / (df['val_accuracy'].max() - df['val_accuracy'].min()), 
                   label='Validation Acc (normalized)', marker='s')
    axes[1, 1].set_title('Normalized Progress')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Normalized Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Moving averages
    window = min(5, len(df) // 3)  # Adaptive window size
    if window >= 2:
        df['acc_ma'] = df['accuracy'].rolling(window=window).mean()
        df['val_acc_ma'] = df['val_accuracy'].rolling(window=window).mean()
        
        axes[1, 2].plot(df['epoch'], df['acc_ma'], label=f'Training MA({window})', linewidth=2)
        axes[1, 2].plot(df['epoch'], df['val_acc_ma'], label=f'Validation MA({window})', linewidth=2)
        axes[1, 2].set_title('Moving Average Accuracy')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage example
if __name__ == "__main__":
    # Replace with your actual file path
    csv_file_path = "training_metrics_resnet3D_augment.csv"
    
    # Basic visualization
    visualize_training_metrics(csv_file_path)
    
    # Advanced analysis
    create_advanced_plots(csv_file_path)
    
    # Optional: Save plots
    # plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')