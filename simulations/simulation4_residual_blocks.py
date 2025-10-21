"""
Simulation 4: Implementing a "Toy" Residual Block
Objective: To understand the "skip connection" or "identity shortcut" that is the core innovation of ResNet.

This simulation compares standard convolutional blocks vs residual blocks to demonstrate:
1. How skip connections help mitigate vanishing gradients
2. How residual networks train more smoothly
3. The concept of identity mapping in deep networks
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class ResidualBlockExperiment:
    def __init__(self, input_shape=(32, 32, 3), num_blocks=10):
        """
        Initialize the experiment with given parameters
        
        Args:
            input_shape: Shape of input images
            num_blocks: Number of blocks to stack in each network
        """
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.results = {}
        
    def create_standard_block(self, x, filters=64, block_name="standard"):
        """
        Create a standard convolutional block without skip connections
        
        Args:
            x: Input tensor
            filters: Number of filters for convolution
            block_name: Name prefix for layers
            
        Returns:
            Output tensor after standard block operations
        """
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same', 
                         name=f'{block_name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                         name=f'{block_name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
        return x
    
    def create_residual_block(self, x, filters=64, block_name="residual"):
        """
        Create a residual block with skip connection
        
        Args:
            x: Input tensor
            filters: Number of filters for convolution
            block_name: Name prefix for layers
            
        Returns:
            Output tensor after residual block operations
        """
        # Store the input for skip connection
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, (3, 3), padding='same', 
                         name=f'{block_name}_conv1')(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
        x = layers.Activation('relu', name=f'{block_name}_relu1')(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same',
                         name=f'{block_name}_conv2')(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
        
        # Skip connection - add the input to the output
        x = layers.Add(name=f'{block_name}_add')([x, shortcut])
        x = layers.Activation('relu', name=f'{block_name}_relu2')(x)
        
        return x
    
    def build_standard_network(self):
        """
        Build a deep network using standard convolutional blocks
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape, name='input')
        x = inputs
        
        # Initial convolution
        x = layers.Conv2D(64, (7, 7), padding='same', activation='relu', name='initial_conv')(x)
        x = layers.BatchNormalization(name='initial_bn')(x)
        
        # Stack multiple standard blocks
        for i in range(self.num_blocks):
            x = self.create_standard_block(x, filters=64, block_name=f'standard_block_{i}')
            
        # Global average pooling and final dense layer
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dense(128, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.5, name='dropout')(x)
        outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
        
        model = keras.Model(inputs, outputs, name='standard_network')
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_residual_network(self):
        """
        Build a deep network using residual blocks
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape, name='input')
        x = inputs
        
        # Initial convolution
        x = layers.Conv2D(64, (7, 7), padding='same', name='initial_conv')(x)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.Activation('relu', name='initial_relu')(x)
        
        # Stack multiple residual blocks
        for i in range(self.num_blocks):
            x = self.create_residual_block(x, filters=64, block_name=f'residual_block_{i}')
            
        # Global average pooling and final dense layer
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dense(128, activation='relu', name='dense1')(x)
        x = layers.Dropout(0.5, name='dropout')(x)
        outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
        
        model = keras.Model(inputs, outputs, name='residual_network')
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_synthetic_data(self, num_samples=1000):
        """
        Generate synthetic data for training the networks
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        # Generate random images
        X = np.random.rand(num_samples, *self.input_shape).astype(np.float32)
        
        # Generate random labels (10 classes)
        y = np.random.randint(0, 10, num_samples)
        
        # Split into train and validation
        split_idx = int(0.8 * num_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    def train_and_compare_models(self, epochs=20, batch_size=32):
        """
        Train both standard and residual networks and compare their performance
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training histories and models
        """
        print("=" * 60)
        print("SIMULATION 4: RESIDUAL BLOCK EXPERIMENT")
        print("=" * 60)
        
        # Generate data
        print("Generating synthetic training data...")
        X_train, y_train, X_val, y_val = self.generate_synthetic_data(num_samples=2000)
        
        # Build models
        print("Building standard network...")
        standard_model = self.build_standard_network()
        
        print("Building residual network...")
        residual_model = self.build_residual_network()
        
        # Print model summaries
        print("\\nStandard Network Architecture:")
        print(f"Total parameters: {standard_model.count_params():,}")
        
        print("\\nResidual Network Architecture:")
        print(f"Total parameters: {residual_model.count_params():,}")
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        # Train standard network
        print("\\nTraining Standard Network...")
        standard_history = standard_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train residual network
        print("\\nTraining Residual Network...")
        residual_history = residual_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store results
        self.results = {
            'standard_model': standard_model,
            'residual_model': residual_model,
            'standard_history': standard_history,
            'residual_history': residual_history,
            'X_val': X_val,
            'y_val': y_val
        }
        
        return self.results
    
    def plot_training_comparison(self, save_path="../results"):
        """
        Plot training curves comparing standard vs residual networks
        
        Args:
            save_path: Directory to save plots
        """
        if not self.results:
            raise ValueError("No training results found. Run train_and_compare_models() first.")
        
        # Create results directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Standard vs Residual Networks Training Comparison', fontsize=16, fontweight='bold')
        
        # Training Loss
        axes[0, 0].plot(self.results['standard_history'].history['loss'], 
                       label='Standard Network', linewidth=2, color='red')
        axes[0, 0].plot(self.results['residual_history'].history['loss'], 
                       label='Residual Network', linewidth=2, color='blue')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation Loss
        axes[0, 1].plot(self.results['standard_history'].history['val_loss'], 
                       label='Standard Network', linewidth=2, color='red')
        axes[0, 1].plot(self.results['residual_history'].history['val_loss'], 
                       label='Residual Network', linewidth=2, color='blue')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training Accuracy
        axes[1, 0].plot(self.results['standard_history'].history['accuracy'], 
                       label='Standard Network', linewidth=2, color='red')
        axes[1, 0].plot(self.results['residual_history'].history['accuracy'], 
                       label='Residual Network', linewidth=2, color='blue')
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation Accuracy
        axes[1, 1].plot(self.results['standard_history'].history['val_accuracy'], 
                       label='Standard Network', linewidth=2, color='red')
        axes[1, 1].plot(self.results['residual_history'].history['val_accuracy'], 
                       label='Residual Network', linewidth=2, color='blue')
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/simulation4_training_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print final results
        print("\\n" + "=" * 60)
        print("FINAL RESULTS COMPARISON")
        print("=" * 60)
        
        # Get final metrics
        std_final_loss = self.results['standard_history'].history['val_loss'][-1]
        res_final_loss = self.results['residual_history'].history['val_loss'][-1]
        std_final_acc = self.results['standard_history'].history['val_accuracy'][-1]
        res_final_acc = self.results['residual_history'].history['val_accuracy'][-1]
        
        print(f"Standard Network - Final Validation Loss: {std_final_loss:.4f}")
        print(f"Residual Network - Final Validation Loss: {res_final_loss:.4f}")
        print(f"Improvement: {((std_final_loss - res_final_loss) / std_final_loss * 100):.2f}%")
        print()
        print(f"Standard Network - Final Validation Accuracy: {std_final_acc:.4f}")
        print(f"Residual Network - Final Validation Accuracy: {res_final_acc:.4f}")
        print(f"Improvement: {((res_final_acc - std_final_acc) / std_final_acc * 100):.2f}%")
        
    def analyze_gradient_flow(self, save_path="../results"):
        """
        Analyze and visualize gradient flow in both networks
        
        Args:
            save_path: Directory to save plots
        """
        if not self.results:
            raise ValueError("No training results found. Run train_and_compare_models() first.")
        
        print("\\n" + "=" * 60)
        print("GRADIENT FLOW ANALYSIS")
        print("=" * 60)
        
        # Create a small batch for gradient analysis
        X_sample = self.results['X_val'][:10]
        y_sample = self.results['y_val'][:10]
        
        # Function to compute gradients
        def compute_gradients(model, x, y):
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = keras.losses.sparse_categorical_crossentropy(y, predictions)
                loss = tf.reduce_mean(loss)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            return gradients
        
        # Compute gradients for both models
        std_gradients = compute_gradients(self.results['standard_model'], X_sample, y_sample)
        res_gradients = compute_gradients(self.results['residual_model'], X_sample, y_sample)
        
        # Calculate gradient magnitudes for each layer
        std_grad_magnitudes = [tf.norm(grad).numpy() if grad is not None else 0 for grad in std_gradients]
        res_grad_magnitudes = [tf.norm(grad).numpy() if grad is not None else 0 for grad in res_gradients]
        
        # Plot gradient magnitudes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Standard network gradients
        ax1.bar(range(len(std_grad_magnitudes)), std_grad_magnitudes, alpha=0.7, color='red')
        ax1.set_title('Gradient Magnitudes - Standard Network')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Gradient Magnitude')
        ax1.set_yscale('log')
        
        # Residual network gradients
        ax2.bar(range(len(res_grad_magnitudes)), res_grad_magnitudes, alpha=0.7, color='blue')
        ax2.set_title('Gradient Magnitudes - Residual Network')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Gradient Magnitude')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/simulation4_gradient_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print analysis
        avg_std_grad = np.mean(std_grad_magnitudes)
        avg_res_grad = np.mean(res_grad_magnitudes)
        
        print(f"Average gradient magnitude - Standard Network: {avg_std_grad:.6f}")
        print(f"Average gradient magnitude - Residual Network: {avg_res_grad:.6f}")
        print(f"Ratio (Residual/Standard): {avg_res_grad/avg_std_grad:.2f}")
        
        if avg_res_grad > avg_std_grad:
            print("✓ Residual network maintains stronger gradients (better gradient flow)")
        else:
            print("⚠ Standard network has stronger gradients in this case")

def main():
    """
    Main function to run the complete Simulation 4 experiment
    """
    print("Starting ResNet Simulation 4: Residual Block Experiment")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize experiment
    experiment = ResidualBlockExperiment(input_shape=(32, 32, 3), num_blocks=8)
    
    # Run training comparison
    results = experiment.train_and_compare_models(epochs=15, batch_size=32)
    
    # Generate visualizations and analysis
    experiment.plot_training_comparison()
    experiment.analyze_gradient_flow()
    
    print("\\n" + "=" * 60)
    print("KEY CONCEPTS DEMONSTRATED:")
    print("=" * 60)
    print("1. Skip Connections: Residual blocks provide direct gradient paths")
    print("2. Vanishing Gradients: Residual networks maintain better gradient flow")
    print("3. Identity Mapping: Network can learn identity functions more easily")
    print("4. Training Stability: Residual networks often train more smoothly")
    print("\\nSimulation 4 completed successfully!")

if __name__ == "__main__":
    main()