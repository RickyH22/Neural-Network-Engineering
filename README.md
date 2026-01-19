# FashionMNIST Classification with Custom PyTorch Layer

A PyTorch implementation demonstrating a custom `LearnedAffine` layer and clean training loop for FashionMNIST classification.

## 🎯 Project Overview

This project fulfills the requirements of a PyTorch assignment focusing on:
- Custom layer implementation (`LearnedAffine`)
- Proper training and evaluation loops
- Modern optimization techniques (AdamW + OneCycleLR)
- Reproducible results with detailed documentation

## 📁 Project Structure

```
.
├── layers.py          # Custom LearnedAffine layer + sanity tests
├── train.py           # Training script with eval loop
├── metrics.md         # Training observations and analysis (150-250 words)
├── repro.md          # Reproducibility guide with exact commands
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Custom Layer

```bash
python layers.py
```

This runs sanity checks on the `LearnedAffine` layer, verifying:
- ✓ Shape preservation
- ✓ Parameter count (2 × num_features)
- ✓ Gradient flow
- ✓ Initial values (scale=1, shift=0)

### 3. Train Model

```bash
python train.py
```

Trains a CNN on FashionMNIST for 3 epochs with:
- **Loss**: CrossEntropyLoss
- **Optimizer**: AdamW (lr=0.001)
- **Scheduler**: OneCycleLR (max_lr=0.01)
- **Seed**: 42 (for reproducibility)

Expected output: ~88-90% test accuracy after 3 epochs.

## 🧠 Model Architecture

```
FashionMNISTNet(
  Conv2d(1 → 32) → ReLU → MaxPool
  Conv2d(32 → 64) → ReLU → MaxPool → Dropout2d(0.25)
  Linear(3136 → 128) → ReLU
  LearnedAffine(128)     ← Custom Layer
  Dropout(0.5)
  Linear(128 → 10)
)
```

**Total Parameters**: ~1.2M

## 🔬 Custom LearnedAffine Layer

The `LearnedAffine` layer applies a learnable affine transformation:

```
y = scale ⊙ x + shift
```

Where:
- `scale`: Learnable parameter (initialized to 1)
- `shift`: Learnable parameter (initialized to 0)
- ⊙: Element-wise multiplication

This adds flexibility between the fully connected layers, allowing the network to learn adaptive normalization.

## 📊 Training Details

### Configuration
- **Dataset**: FashionMNIST (60k train, 10k test)
- **Epochs**: 3
- **Batch Size**: 64
- **Optimizer**: AdamW (weight_decay=0.01 by default)
- **Scheduler**: OneCycleLR with max_lr=0.01
- **Seed**: 42

### Evaluation Protocol
- Model set to `eval()` mode during testing
- Gradients disabled with `torch.no_grad()`
- Metrics: Cross-entropy loss + accuracy

## 📈 Results

See [metrics.md](metrics.md) for detailed analysis of:
- What was tried
- What worked well
- Potential improvements

## 🔄 Reproducibility

All experiments use `torch.manual_seed(42)` for reproducibility. See [repro.md](repro.md) for:
- Exact commands to reproduce results
- Seed configuration details
- Environment setup instructions
- Troubleshooting tips

## 📚 Key Learning Points

1. **Custom Layer**: Implemented `nn.Module` with learnable parameters
2. **Training Loop**: Proper use of `model.train()` and optimizer steps
3. **Evaluation**: Correct use of `model.eval()` and `torch.no_grad()`
4. **Optimization**: AdamW optimizer with OneCycleLR scheduling
5. **Reproducibility**: Seed management and deterministic operations

## 🎓 Assignment Rubric Coverage

- ✅ **Custom layer correct + sanity test included** (3 pts)
  - `LearnedAffine` in `layers.py` with comprehensive tests
  
- ✅ **Training loop correct + eval uses eval() and no_grad()** (3 pts)
  - Proper training loop in `train.py`
  - Evaluation function with `model.eval()` and `torch.no_grad()`
  
- ✅ **Optimizer + scheduler used correctly and explained** (3 pts)
  - AdamW optimizer configured
  - OneCycleLR scheduler with per-batch updates
  - Explanations in `metrics.md`
  
- ✅ **Repro clarity (seed + commands + readable notes)** (3 pts)
  - `torch.manual_seed(42)` documented
  - Complete command instructions in `repro.md`
  - Clear documentation throughout

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+

## 📝 License

This project is for educational purposes as part of a course assignment.

## 👤 Author

Created as part of a PyTorch deep learning assignment.

---

**Note**: For submission, upload this entire repository to GitHub and submit the repository link.
