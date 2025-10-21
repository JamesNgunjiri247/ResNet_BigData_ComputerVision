"""
Data Utilities for ResNet and Big Data Computer Vision Simulations
This module provides data loading, preprocessing, and augmentation utilities
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile
from PIL import Image

class DataUtils:
    """
    Utility class for data handling and preprocessing
    """
    
    @staticmethod
    def create_synthetic_dataset(num_samples=1000, img_size=(32, 32), num_classes=10, noise_level=0.1):
        """
        Create synthetic dataset for testing
        
        Args:
            num_samples: Total number of samples
            img_size: Size of generated images
            num_classes: Number of classes
            noise_level: Amount of noise to add
            
        Returns:
            Tuple of (X, y) arrays
        """
        X = np.random.rand(num_samples, *img_size, 3).astype(np.float32)
        y = np.random.randint(0, num_classes, num_samples)
        
        # Add class-specific patterns
        for class_idx in range(num_classes):
            class_mask = y == class_idx
            class_samples = np.sum(class_mask)
            
            # Add unique pattern for each class
            pattern = np.sin(np.linspace(0, 2*np.pi*class_idx, img_size[0]))
            pattern = np.tile(pattern, (class_samples, img_size[1], 3, 1))
            pattern = np.transpose(pattern, (0, 3, 1, 2))
            
            X[class_mask] += pattern * 0.3
        
        # Add noise
        X += np.random.normal(0, noise_level, X.shape)
        X = np.clip(X, 0, 1)
        
        return X, y
    
    @staticmethod
    def preprocess_for_resnet(images, target_size=(224, 224)):
        """
        Preprocess images for ResNet models
        
        Args:
            images: Input images array
            target_size: Target image size
            
        Returns:
            Preprocessed images
        """
        # Resize images
        if images.shape[1:3] != target_size:
            resized_images = []
            for img in images:
                resized = cv2.resize(img, target_size)
                resized_images.append(resized)
            images = np.array(resized_images)
        
        # Apply ResNet preprocessing
        images = keras.applications.resnet50.preprocess_input(images * 255.0)
        
        return images
    
    @staticmethod
    def create_data_augmentation():
        """
        Create data augmentation pipeline
        
        Returns:
            ImageDataGenerator for augmentation
        """
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        return datagen
    
    @staticmethod
    def split_dataset(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            X: Features
            y: Labels
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=random_state
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size), stratify=y_temp, random_state=random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

class ModelUtils:
    """
    Utility class for model operations
    """
    
    @staticmethod
    def count_parameters(model):
        """
        Count total and trainable parameters in a model
        
        Args:
            model: Keras model
            
        Returns:
            Dictionary with parameter counts
        """
        total_params = model.count_params()
        trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
        non_trainable_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    @staticmethod
    def get_layer_info(model):
        """
        Get detailed information about model layers
        
        Args:
            model: Keras model
            
        Returns:
            List of layer information dictionaries
        """
        layer_info = []
        
        for i, layer in enumerate(model.layers):
            info = {
                'index': i,
                'name': layer.name,
                'type': type(layer).__name__,
                'output_shape': layer.output_shape,
                'trainable': layer.trainable,
                'params': layer.count_params()
            }
            layer_info.append(info)
        
        return layer_info
    
    @staticmethod
    def freeze_layers(model, freeze_until_layer=None, freeze_last_n=None):
        """
        Freeze layers in a model
        
        Args:
            model: Keras model
            freeze_until_layer: Freeze layers up to this layer name/index
            freeze_last_n: Freeze the last n layers
            
        Returns:
            Modified model
        """
        if freeze_until_layer is not None:
            if isinstance(freeze_until_layer, str):
                # Find layer by name
                freeze_index = None
                for i, layer in enumerate(model.layers):
                    if layer.name == freeze_until_layer:
                        freeze_index = i
                        break
            else:
                freeze_index = freeze_until_layer
            
            if freeze_index is not None:
                for i in range(freeze_index + 1):
                    model.layers[i].trainable = False
        
        if freeze_last_n is not None:
            for i in range(len(model.layers) - freeze_last_n, len(model.layers)):
                model.layers[i].trainable = False
        
        return model
    
    @staticmethod
    def unfreeze_layers(model, unfreeze_from_layer=None, unfreeze_last_n=None):
        """
        Unfreeze layers in a model
        
        Args:
            model: Keras model
            unfreeze_from_layer: Unfreeze layers from this layer name/index
            unfreeze_last_n: Unfreeze the last n layers
            
        Returns:
            Modified model
        """
        if unfreeze_from_layer is not None:
            if isinstance(unfreeze_from_layer, str):
                # Find layer by name
                unfreeze_index = None
                for i, layer in enumerate(model.layers):
                    if layer.name == unfreeze_from_layer:
                        unfreeze_index = i
                        break
            else:
                unfreeze_index = unfreeze_from_layer
            
            if unfreeze_index is not None:
                for i in range(unfreeze_index, len(model.layers)):
                    model.layers[i].trainable = True
        
        if unfreeze_last_n is not None:
            for i in range(len(model.layers) - unfreeze_last_n, len(model.layers)):
                model.layers[i].trainable = True
        
        return model

class ExperimentUtils:
    """
    Utility class for managing experiments
    """
    
    @staticmethod
    def setup_experiment_directory(base_dir="../results", experiment_name=None):
        """
        Set up directory structure for experiments
        
        Args:
            base_dir: Base directory for results
            experiment_name: Name of the experiment
            
        Returns:
            Path to experiment directory
        """
        if experiment_name is None:
            from datetime import datetime
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        exp_dir = os.path.join(base_dir, experiment_name)
        
        # Create subdirectories
        subdirs = ['models', 'plots', 'logs', 'data']
        for subdir in subdirs:
            os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
        
        return exp_dir
    
    @staticmethod
    def save_experiment_config(config, experiment_dir):
        """
        Save experiment configuration
        
        Args:
            config: Configuration dictionary
            experiment_dir: Experiment directory
        """
        import json
        
        config_path = os.path.join(experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {config_path}")
    
    @staticmethod
    def create_callbacks(experiment_dir, patience=5, save_best_only=True):
        """
        Create standard callbacks for training
        
        Args:
            experiment_dir: Experiment directory
            patience: Patience for early stopping
            save_best_only: Whether to save only the best model
            
        Returns:
            List of callbacks
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(experiment_dir, 'models', 'best_model.h5'),
                save_best_only=save_best_only,
                save_weights_only=False,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                filename=os.path.join(experiment_dir, 'logs', 'training.log'),
                append=True
            )
        ]
        
        return callbacks

# Example usage
if __name__ == "__main__":
    print("Testing Data Utilities...")
    
    # Test synthetic dataset creation
    X, y = DataUtils.create_synthetic_dataset(num_samples=100, img_size=(64, 64), num_classes=5)
    print(f"Created synthetic dataset: X shape {X.shape}, y shape {y.shape}")
    
    # Test data splitting
    X_train, X_val, X_test, y_train, y_val, y_test = DataUtils.split_dataset(X, y)
    print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Test experiment setup
    exp_dir = ExperimentUtils.setup_experiment_directory(experiment_name="test_experiment")
    print(f"Experiment directory created: {exp_dir}")
    
    print("All utilities tested successfully!")