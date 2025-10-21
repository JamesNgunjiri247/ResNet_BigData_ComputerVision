"""
Quick Test Script to Generate Sample Results
This script runs a simplified version of the ResNet simulations to demonstrate the concepts
with faster execution and smaller models.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style and suppress warnings
plt.style.use('default')
sns.set_palette("husl")
tf.get_logger().setLevel('ERROR')

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

def create_simple_residual_block(x, filters=32):
    """Create a simple residual block"""
    shortcut = x
    
    # Main path
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def create_simple_standard_block(x, filters=32):
    """Create a simple standard block (no skip connection)"""
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def build_sample_models():
    """Build simple sample models for comparison"""
    input_shape = (32, 32, 3)
    
    # Standard Network
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    
    # Add 3 standard blocks
    for i in range(3):
        x = create_simple_standard_block(x, 32)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    standard_model = keras.Model(inputs, outputs, name='standard_cnn')
    
    # Residual Network
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    
    # Add 3 residual blocks
    for i in range(3):
        x = create_simple_residual_block(x, 32)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    residual_model = keras.Model(inputs, outputs, name='residual_cnn')
    
    return standard_model, residual_model

def generate_sample_data(num_samples=500):
    """Generate synthetic data for quick testing"""
    X = np.random.rand(num_samples, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, 10, num_samples)
    
    # Add some class-specific patterns
    for class_idx in range(10):
        class_mask = y == class_idx
        # Add slight pattern for each class
        X[class_mask, :, :, 0] += (class_idx / 10) * 0.2
    
    return X, y

def train_and_compare():
    """Train both models and generate comparison results"""
    print("=" * 60)
    print("QUICK RESNET SIMULATION DEMO")
    print("=" * 60)
    
    # Create models
    print("Building models...")
    standard_model, residual_model = build_sample_models()
    
    # Compile models
    for model in [standard_model, residual_model]:
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    print(f"Standard model parameters: {standard_model.count_params():,}")
    print(f"Residual model parameters: {residual_model.count_params():,}")
    
    # Generate data
    print("\\nGenerating training data...")
    X, y = generate_sample_data(1000)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train models
    epochs = 5  # Quick training
    print(f"\\nTraining for {epochs} epochs...")
    
    # Train standard model
    print("Training standard model...")
    std_history = standard_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    
    # Train residual model
    print("\\nTraining residual model...")
    res_history = residual_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    
    return std_history, res_history, X_val, y_val

def plot_results(std_history, res_history, save_dir="../results"):
    """Generate comparison plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Quick ResNet Demo: Standard vs Residual Networks', fontsize=14, fontweight='bold')
    
    # Training Loss
    axes[0, 0].plot(std_history.history['loss'], label='Standard CNN', color='red', linewidth=2)
    axes[0, 0].plot(res_history.history['loss'], label='Residual CNN', color='blue', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation Loss
    axes[0, 1].plot(std_history.history['val_loss'], label='Standard CNN', color='red', linewidth=2)
    axes[0, 1].plot(res_history.history['val_loss'], label='Residual CNN', color='blue', linewidth=2)
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Accuracy
    axes[1, 0].plot(std_history.history['accuracy'], label='Standard CNN', color='red', linewidth=2)
    axes[1, 0].plot(res_history.history['accuracy'], label='Residual CNN', color='blue', linewidth=2)
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation Accuracy
    axes[1, 1].plot(std_history.history['val_accuracy'], label='Standard CNN', color='red', linewidth=2)
    axes[1, 1].plot(res_history.history['val_accuracy'], label='Residual CNN', color='blue', linewidth=2)
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/quick_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    std_final_acc = std_history.history['val_accuracy'][-1]
    res_final_acc = res_history.history['val_accuracy'][-1]
    
    print(f"Standard CNN - Final Validation Accuracy: {std_final_acc:.4f}")
    print(f"Residual CNN - Final Validation Accuracy: {res_final_acc:.4f}")
    
    if res_final_acc > std_final_acc:
        improvement = ((res_final_acc - std_final_acc) / std_final_acc) * 100
        print(f"✅ Residual network performed {improvement:.2f}% better!")
    else:
        print("⚠️ In this quick demo, standard network performed better (try more epochs)")
    
    print(f"\\n📁 Results saved to: {save_dir}/quick_demo_results.png")

def transfer_learning_demo():
    """Quick transfer learning demonstration"""
    print("\\n" + "=" * 60)
    print("QUICK TRANSFER LEARNING DEMO")
    print("=" * 60)
    
    # Load a pre-trained model (smaller for demo)
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(96, 96, 3)  # Smaller input for speed
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add classifier
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Transfer learning model parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(var).numpy() for var in model.trainable_variables]):,}")
    
    # Generate some sample data
    X_sample = np.random.rand(200, 96, 96, 3)
    y_sample = np.random.randint(0, 2, 200)
    
    print("Training transfer learning model...")
    history = model.fit(X_sample, y_sample, epochs=3, validation_split=0.2, verbose=1)
    
    final_acc = history.history['val_accuracy'][-1]
    print(f"\\nTransfer learning final accuracy: {final_acc:.4f}")
    print("✅ Transfer learning demo completed!")
    
    return history

def main():
    """Main function to run the quick demo"""
    print(f"🚀 Quick ResNet Demo Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Run residual vs standard comparison
    std_history, res_history, X_val, y_val = train_and_compare()
    
    # Generate plots
    plot_results(std_history, res_history)
    
    # Run transfer learning demo
    transfer_learning_demo()
    
    print("\\n" + "=" * 60)
    print("KEY TAKEAWAYS FROM QUICK DEMO")
    print("=" * 60)
    print("1. 🔗 Residual connections help with gradient flow")
    print("2. 📈 Skip connections often lead to better training")
    print("3. 🎯 Transfer learning leverages pre-trained features")
    print("4. ⚡ Pre-trained models require fewer trainable parameters")
    print("\\n✅ Quick demo completed successfully!")
    print("\\n📋 For full simulations with detailed analysis:")
    print("   python simulations/simulation4_residual_blocks.py")
    print("   python simulations/simulation5_transfer_learning.py")

if __name__ == "__main__":
    main()