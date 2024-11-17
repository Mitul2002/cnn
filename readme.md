# Pneumonia Detection with CNNs

A deep learning system for pneumonia detection in chest X-rays using Convolutional Neural Networks, optimized for GPU inference and achieving 92% accuracy.

## Overview

This project implements a robust CNN architecture for detecting pneumonia from chest X-ray images. It features transfer learning with custom optimization layers and efficient GPU acceleration techniques.

## Model Architecture

```python
ConvBlock Architecture:
- Input Layer (3 channels)
- Conv Block 1: 256 filters (4x4), SELU, MaxPool(3)
- Conv Block 2: 512 filters (3x3), SELU, MaxPool(2)
- Conv Block 3: 1024 filters (2x2), SELU, MaxPool(2)
- Global Average Pooling
- Dense Layer (2 units)
```

## Key Features

- Custom ConvBlock implementation with SELU activation
- GPU-optimized inference pipeline
- Model quantization and pruning
- Batch processing support
- Early stopping and model checkpointing
- Mixed precision training

## Performance Metrics

- Accuracy: 92% on test set
- AUC-ROC Score: 0.91
- Inference Time: 25% reduction through optimization
- Memory Footprint: Reduced by 30% through quantization

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Run inference
python predict.py --image path/to/xray.jpg
```

## Dataset

The model is trained on chest X-ray images with two classes:
- Normal
- Pneumonia

Dataset should be organized as:
```
data/
├── train/
│   ├── normal/
│   └── pneumonia/
└── test/
    ├── normal/
    └── pneumonia/
```

## Model Training

```python
# Example training code
model = DeepCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Train with early stopping and validation
train(model, train_loader, val_loader, 
      optimizer, criterion, num_epochs=32)
```

## Contributors

- Mitul Solanki

## License

MIT License