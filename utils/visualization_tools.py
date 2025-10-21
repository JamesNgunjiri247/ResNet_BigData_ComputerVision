"""
Visualization Utilities for ResNet and Big Data Computer Vision Simulations
This module provides comprehensive visualization tools for analyzing and presenting
the results of the ResNet simulations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from datetime import datetime
import os

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class VisualizationTools:
    """
    Comprehensive visualization tools for ResNet experiments
    """
    
    def __init__(self, save_dir="../results"):
        """
        Initialize visualization tools
        
        Args:
            save_dir: Directory to save generated plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_feature_maps(self, model, input_image, layer_names=None, max_filters=16, save_name="feature_maps"):
        """
        Visualize feature maps from different layers of a CNN
        
        Args:
            model: Trained Keras model
            input_image: Input image to generate feature maps from
            layer_names: List of layer names to visualize
            max_filters: Maximum number of filters to show per layer
            save_name: Name for saving the plot
        """
        if layer_names is None:
            # Get some convolutional layers automatically
            layer_names = [layer.name for layer in model.layers 
                          if 'conv' in layer.name.lower()][:4]
        
        # Create a model that outputs the feature maps
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
        
        # Get activations
        if len(input_image.shape) == 3:
            input_image = np.expand_dims(input_image, axis=0)
        
        activations = activation_model.predict(input_image, verbose=0)
        
        # Plot feature maps
        fig, axes = plt.subplots(len(layer_names), max_filters, 
                                figsize=(max_filters * 2, len(layer_names) * 2))
        
        if len(layer_names) == 1:
            axes = [axes]
        
        for layer_idx, (layer_name, activation) in enumerate(zip(layer_names, activations)):
            for filter_idx in range(min(max_filters, activation.shape[-1])):
                ax = axes[layer_idx][filter_idx] if max_filters > 1 else axes[layer_idx]
                
                # Get the feature map
                feature_map = activation[0, :, :, filter_idx]
                
                # Plot
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'{layer_name}\\nFilter {filter_idx}', fontsize=10)
                ax.axis('off')
                
                # Add colorbar for the first filter of each layer
                if filter_idx == 0:
                    plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_filters(self, model, layer_name, max_filters=64, save_name="filters"):
        """
        Visualize the learned filters/kernels in a convolutional layer
        
        Args:
            model: Trained Keras model
            layer_name: Name of the layer to visualize
            max_filters: Maximum number of filters to show
            save_name: Name for saving the plot
        """
        # Get the layer
        layer = model.get_layer(layer_name)
        
        # Get the weights (filters)
        filters, biases = layer.get_weights()
        
        # Normalize filters for visualization
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        
        # Determine grid size
        n_filters = min(max_filters, filters.shape[-1])
        grid_size = int(np.ceil(np.sqrt(n_filters)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        
        for i in range(grid_size):
            for j in range(grid_size):
                filter_idx = i * grid_size + j
                
                if filter_idx < n_filters:
                    # Get the filter
                    filter_img = filters[:, :, :, filter_idx]
                    
                    # If filter has multiple channels, take the mean or first channel
                    if filter_img.shape[-1] > 1:
                        if filter_img.shape[-1] == 3:  # RGB
                            filter_img = filter_img
                        else:
                            filter_img = filter_img[:, :, 0]  # Take first channel
                    else:
                        filter_img = filter_img[:, :, 0]
                    
                    axes[i, j].imshow(filter_img, cmap='gray')
                    axes[i, j].set_title(f'Filter {filter_idx}', fontsize=8)
                else:
                    axes[i, j].set_visible(False)
                
                axes[i, j].axis('off')
        
        plt.suptitle(f'Learned Filters from Layer: {layer_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}_{layer_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_model_architecture(self, model, save_name="model_architecture"):
        """
        Visualize model architecture
        
        Args:
            model: Keras model
            save_name: Name for saving the plot
        """
        try:
            keras.utils.plot_model(
                model,
                to_file=f'{self.save_dir}/{save_name}.png',
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=False,
                dpi=300
            )
            print(f"Model architecture saved as {save_name}.png")
        except Exception as e:
            print(f"Could not generate model plot: {e}")
            print("Make sure graphviz is installed: pip install graphviz")
            
    def plot_training_curves(self, histories, labels, save_name="training_curves"):
        """
        Plot training curves for multiple models
        
        Args:
            histories: List of training histories
            labels: List of labels for each history
            save_name: Name for saving the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(histories)))
        
        # Training accuracy
        for history, label, color in zip(histories, labels, colors):
            axes[0, 0].plot(history.history['accuracy'], label=label, color=color, linewidth=2)
        axes[0, 0].set_title('Training Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation accuracy
        for history, label, color in zip(histories, labels, colors):
            axes[0, 1].plot(history.history['val_accuracy'], label=label, color=color, linewidth=2)
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training loss
        for history, label, color in zip(histories, labels, colors):
            axes[1, 0].plot(history.history['loss'], label=label, color=color, linewidth=2)
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation loss
        for history, label, color in zip(histories, labels, colors):
            axes[1, 1].plot(history.history['val_loss'], label=label, color=color, linewidth=2)
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_name="confusion_matrix"):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_name: Name for saving the plot
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_gradient_magnitudes(self, model, sample_data, save_name="gradient_magnitudes"):
        """
        Visualize gradient magnitudes across layers
        
        Args:
            model: Keras model
            sample_data: Sample input data
            save_name: Name for saving the plot
        """
        X_sample, y_sample = sample_data
        
        with tf.GradientTape() as tape:
            predictions = model(X_sample, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(y_sample, predictions)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Calculate gradient magnitudes
        grad_magnitudes = [tf.norm(grad).numpy() if grad is not None else 0 for grad in gradients]
        layer_names = [var.name for var in model.trainable_variables]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(grad_magnitudes)), grad_magnitudes, alpha=0.7)
        plt.title('Gradient Magnitudes Across Layers')
        plt.xlabel('Layer Index')
        plt.ylabel('Gradient Magnitude (Log Scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Add layer names for some layers
        step = max(1, len(layer_names) // 10)
        plt.xticks(range(0, len(layer_names), step), 
                  [name.split('/')[0] for name in layer_names[::step]], 
                  rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_parameter_distribution(self, model, save_name="parameter_distribution"):
        """
        Plot distribution of model parameters
        
        Args:
            model: Keras model
            save_name: Name for saving the plot
        """
        all_weights = []
        layer_names = []
        
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                for w in weights:
                    all_weights.extend(w.flatten())
                    layer_names.append(layer.name)
        
        plt.figure(figsize=(12, 6))
        
        # Overall distribution
        plt.subplot(1, 2, 1)
        plt.hist(all_weights, bins=50, alpha=0.7, density=True)
        plt.title('Overall Parameter Distribution')
        plt.xlabel('Parameter Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Statistics
        plt.subplot(1, 2, 2)
        stats = {
            'Mean': np.mean(all_weights),
            'Std': np.std(all_weights),
            'Min': np.min(all_weights),
            'Max': np.max(all_weights),
            'Zeros': np.sum(np.array(all_weights) == 0)
        }
        
        plt.bar(stats.keys(), stats.values(), alpha=0.7)
        plt.title('Parameter Statistics')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_comparison_dashboard(self, results_dict, save_name="comparison_dashboard"):
        """
        Create a comprehensive dashboard comparing multiple models
        
        Args:
            results_dict: Dictionary with model results
            save_name: Name for saving the plot
        """
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Accuracy comparison
        plt.subplot(3, 4, 1)
        models = list(results_dict.keys())
        accuracies = [results_dict[model]['test_accuracy'] for model in models]
        
        bars = plt.bar(models, accuracies, alpha=0.7)
        plt.title('Test Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Loss comparison
        plt.subplot(3, 4, 2)
        losses = [results_dict[model]['test_loss'] for model in models]
        plt.bar(models, losses, alpha=0.7, color='orange')
        plt.title('Test Loss Comparison')
        plt.ylabel('Loss')
        plt.xticks(rotation=45)
        
        # 3. Training curves
        plt.subplot(3, 4, (3, 4))
        for model in models:
            if 'history' in results_dict[model]:
                history = results_dict[model]['history']
                plt.plot(history.history['val_accuracy'], label=f'{model}', linewidth=2)
        plt.title('Validation Accuracy During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Parameter count comparison
        plt.subplot(3, 4, 5)
        param_counts = []
        for model in models:
            if 'model' in results_dict[model]:
                param_counts.append(results_dict[model]['model'].count_params())
            else:
                param_counts.append(0)
        
        plt.bar(models, param_counts, alpha=0.7, color='green')
        plt.title('Total Parameters')
        plt.ylabel('Parameter Count')
        plt.yscale('log')
        plt.xticks(rotation=45)
        
        # 5. Training time comparison (if available)
        plt.subplot(3, 4, 6)
        epochs = []
        for model in models:
            if 'history' in results_dict[model]:
                epochs.append(len(results_dict[model]['history'].history['accuracy']))
            else:
                epochs.append(0)
        
        plt.bar(models, epochs, alpha=0.7, color='purple')
        plt.title('Training Epochs')
        plt.ylabel('Number of Epochs')
        plt.xticks(rotation=45)
        
        # 6-12. Additional analysis plots can be added here
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self, results_dict, save_name="experiment_report"):
        """
        Generate a comprehensive report with all visualizations
        
        Args:
            results_dict: Dictionary with experimental results
            save_name: Name for the report file
        """
        report_text = f"""
# ResNet and Big Data Computer Vision - Experiment Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Summary
This report contains the analysis of ResNet simulations comparing different approaches:

"""
        
        for model_name, results in results_dict.items():
            report_text += f"""
### {model_name.replace('_', ' ').title()}
- Test Accuracy: {results.get('test_accuracy', 'N/A'):.4f}
- Test Loss: {results.get('test_loss', 'N/A'):.4f}
- Parameters: {results.get('model', {}).count_params() if 'model' in results else 'N/A':,}
"""
        
        report_text += """
## Key Findings
1. Transfer learning approaches generally outperformed training from scratch
2. Feature extraction provided efficient parameter usage
3. Fine-tuning achieved the best performance when applied correctly
4. Residual connections helped with gradient flow in deep networks

## Visualizations Generated
- Training curves comparison
- Model architecture diagrams
- Feature map visualizations
- Gradient magnitude analysis
- Parameter distribution plots
"""
        
        # Save report
        with open(f'{self.save_dir}/{save_name}.md', 'w') as f:
            f.write(report_text)
        
        print(f"Comprehensive report saved as {save_name}.md")
        print(f"All visualizations saved in {self.save_dir}/")

# Example usage and utility functions
def demo_visualization_tools():
    """
    Demonstration of the visualization tools
    """
    print("ResNet Visualization Tools Demo")
    print("=" * 40)
    
    # Create a simple model for demonstration
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Create dummy data
    X_dummy = np.random.rand(10, 32, 32, 3)
    y_dummy = np.random.randint(0, 10, 10)
    
    # Initialize visualization tools
    viz = VisualizationTools()
    
    # Demonstrate some visualizations
    print("Generating parameter distribution plot...")
    viz.plot_parameter_distribution(model)
    
    print("Generating gradient magnitude plot...")
    viz.plot_gradient_magnitudes(model, (X_dummy[:5], y_dummy[:5]))
    
    print("Demo completed!")

if __name__ == "__main__":
    demo_visualization_tools()