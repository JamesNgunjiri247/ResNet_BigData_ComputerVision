# ResNet and Big Data Computer Vision Simulations

**Course**: BCS-05-0839/2022  
**Presenter**: James Gitari  
**Support Groups**: 7, 8, 9

This project implements comprehensive simulations to understand ResNet architecture and the role of big data in computer vision, focusing on two key simulations:

- **Simulation 4**: Implementing "Toy" Residual Blocks
- **Simulation 5**: Transfer Learning with Pre-trained ResNet on Small Datasets

## 📁 Project Structure

```
ResNet_BigData_ComputerVision/
├── simulations/
│   ├── simulation4_residual_blocks.py      # Residual vs Standard blocks comparison
│   └── simulation5_transfer_learning.py    # Transfer learning experiment
├── utils/
│   ├── __init__.py                         # Package initialization
│   ├── visualization_tools.py              # Plotting and visualization utilities
│   └── data_utils.py                      # Data processing and model utilities
├── results/                               # Generated plots and results
├── data/                                  # Dataset storage
├── requirements.txt                       # Python dependencies
├── README.md                             # This file
├── setup_environment.py                  # Environment setup script
└── run_all_simulations.py               # Master script to run all experiments
```

## 🎯 Learning Objectives

### Simulation 4: Residual Blocks
- Understand **skip connections** and **identity shortcuts**
- Demonstrate how residual blocks mitigate **vanishing gradients**
- Compare training stability between standard and residual networks
- Explore the concept of **identity mapping** in deep networks

### Simulation 5: Transfer Learning
- Demonstrate the power of **big data** (ImageNet) for small dataset problems
- Compare **feature extraction** vs **fine-tuning** approaches
- Show efficiency gains from transfer learning vs training from scratch
- Understand when to use different transfer learning strategies

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.8 or higher
- At least 8GB RAM recommended
- GPU support optional but recommended for faster training

### Step 1: Clone/Download the Project
```bash
# If using git
git clone <repository-url>
cd ResNet_BigData_ComputerVision

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# For GPU support (optional)
pip install tensorflow-gpu
```

### Step 3: Verify Installation
```bash
python setup_environment.py
```

## 🚀 Running the Simulations

### Option 1: Run All Simulations at Once
```bash
python run_all_simulations.py
```

### Option 2: Run Individual Simulations

#### Simulation 4: Residual Blocks
```bash
cd simulations
python simulation4_residual_blocks.py
```

#### Simulation 5: Transfer Learning
```bash
cd simulations
python simulation5_transfer_learning.py
```

### Option 3: Interactive Jupyter Notebook
```bash
jupyter notebook
# Open and run the provided notebooks
```

## 📊 Expected Results

### Simulation 4 Outputs
- **Training curves**: Comparison of loss and accuracy between standard and residual networks
- **Gradient analysis**: Visualization of gradient magnitudes across layers
- **Performance metrics**: Final accuracy and convergence analysis
- **Key finding**: Residual networks typically show smoother training and better gradient flow

### Simulation 5 Outputs
- **Model comparison**: Baseline CNN vs Feature Extractor vs Fine-tuning
- **Performance analysis**: Test accuracy and training efficiency
- **Parameter efficiency**: Comparison of trainable parameters
- **Transfer learning benefits**: Demonstration of big data impact

## 📈 Understanding the Results

### Key Concepts Demonstrated

#### 1. Vanishing Gradient Problem
- **Standard networks**: Gradients diminish as they propagate backward through deep layers
- **Residual networks**: Skip connections provide direct gradient paths, maintaining strong gradients

#### 2. Identity Mapping
- Residual blocks can learn to be identity functions by setting convolutional weights to zero
- This is harder for standard blocks, making deep networks difficult to train

#### 3. Transfer Learning Power
- **Feature Extraction**: Use pre-trained layers as fixed feature extractors
- **Fine-tuning**: Adapt pre-trained features to new tasks
- **Efficiency**: Achieve better results with less data and training time

#### 4. Big Data Impact
- ImageNet pre-training (1.2M images) provides powerful general features
- These features transfer well to specific tasks even with limited data
- Demonstrates how large-scale datasets benefit the entire computer vision community

## 🔍 Analysis Guidelines

When analyzing your results, consider these questions:

### For Simulation 4:
1. **How do training curves differ between standard and residual networks?**
   - Look for smoother convergence in residual networks
   - Compare final accuracy and loss values

2. **What do gradient magnitudes tell us?**
   - Residual networks should maintain stronger gradients in deeper layers
   - Standard networks may show gradient vanishing

3. **Why do residual blocks help with deep networks?**
   - Skip connections provide "gradient highways"
   - Enable training of very deep networks (50+ layers)

### For Simulation 5:
1. **How much does transfer learning improve performance?**
   - Compare test accuracy: Baseline vs Feature Extractor vs Fine-tuning
   - Note the improvement percentages

2. **What's the efficiency gain?**
   - Compare training time and epochs to convergence
   - Look at parameter efficiency (trainable vs total parameters)

3. **When is fine-tuning beneficial?**
   - Fine-tuning helps when new dataset is somewhat similar to ImageNet
   - Risk of overfitting with very small datasets

## 📋 Assignment Deliverables

For each simulation, your report should include:

### 1. Objective
Clear statement of what the simulation aims to demonstrate

### 2. Code
- Well-commented, functional implementation
- Proper use of the provided utilities
- Clear variable names and documentation

### 3. Results
- **Graphs**: Training curves, comparison plots
- **Visualizations**: Feature maps, gradient analysis
- **Quantitative results**: Final accuracy, improvement percentages

### 4. Analysis & Discussion
**This is the most important part!** Explain:
- Why you got the results you did
- What the visualizations tell you about the underlying concepts
- How the results support or contradict theoretical expectations
- Implications for real-world applications

### Example Analysis Questions:
- Why did the residual network train more smoothly?
- How did skip connections affect gradient flow?
- Why did transfer learning outperform training from scratch?
- What does this tell us about the value of big datasets?

## 🛠️ Customization and Extensions

### Modify Experimental Parameters
```python
# In simulation4_residual_blocks.py
experiment = ResidualBlockExperiment(
    input_shape=(64, 64, 3),  # Change image size
    num_blocks=15             # Change network depth
)

# In simulation5_transfer_learning.py
experiment = TransferLearningExperiment(
    img_size=(224, 224),      # ResNet input size
    num_classes=2             # Cats vs Dogs
)
```

### Add Your Own Datasets
```python
# Replace synthetic data with real datasets
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

### Experiment with Different Architectures
```python
# Try different pre-trained models
base_model = keras.applications.VGG16(weights='imagenet', include_top=False)
base_model = keras.applications.InceptionV3(weights='imagenet', include_top=False)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Memory Errors
- Reduce batch size: `batch_size=16` or `batch_size=8`
- Use smaller image sizes
- Reduce number of samples

#### 2. GPU Issues
- Install proper CUDA drivers
- Check GPU memory availability
- Fall back to CPU if needed: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

#### 3. Slow Training
- Reduce number of epochs for testing
- Use smaller datasets
- Consider using Google Colab for GPU access

#### 4. Import Errors
- Ensure all packages are installed: `pip install -r requirements.txt`
- Check Python version compatibility
- Verify TensorFlow installation

### Getting Help

1. Check the error messages carefully
2. Review the comments in the code
3. Try the troubleshooting steps above
4. Consult TensorFlow documentation
5. Ask classmates or instructors

## 📚 Additional Resources

### Theoretical Background
- **ResNet Paper**: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- **Transfer Learning**: "How transferable are features in deep neural networks?" (Yosinski et al., 2014)

### Practical Tutorials
- TensorFlow transfer learning guide
- Keras applications documentation
- Deep learning visualization techniques

### Datasets for Extension
- CIFAR-10/CIFAR-100
- Dogs vs Cats (Kaggle)
- Custom datasets from your own images

## 👥 Group Members

### Group 7
- **BCS-03-0104/2025**: Gakii Kinge Betty
- **BCS-03-0112/2025**: Mburu Mwangi Elton

### Group 8
- **BCS-03-0097/2025**: Mwangi Ngugi Jeffeson
- **BCS-05-0428/2023**: Omukata Indengu

### Group 9
- **BCS-03-0103/2025**: Nkando Kinoti Wisdom
- **BCS-03-0118/2025**: Samuel D.otieno

## 📄 License

This project is for educational purposes as part of the BCS-05-0839/2022 course.

---

**Good luck with your simulations!** 🎓

Remember: The goal is not just to run the code, but to understand the deep learning concepts and be able to explain what you observe. Focus on the analysis and discussion sections of your reports.

For questions or issues, refer to the troubleshooting section or consult with your group members and instructors.