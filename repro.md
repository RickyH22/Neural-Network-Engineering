# Reproducibility Guide

This document contains exact commands and configuration details to reproduce the training results.

## Environment Setup

1. **Python Version**: Python 3.8+ recommended

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision
   ```

   Or use the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```

## Random Seed

The training script uses a fixed random seed for reproducibility:

```python
SEED = 42
torch.manual_seed(SEED)
```

This seed is set at the beginning of `train.py` and ensures deterministic behavior across:
- PyTorch operations
- CUDA operations (if GPU is available)
- cuDNN operations (deterministic mode enabled)

## Training Commands

### 1. Test the Custom Layer

First, verify that the custom `LearnedAffine` layer works correctly:

```bash
python layers.py
```

**Expected Output**: You should see a sanity check report confirming:
- Shape preservation (input/output shapes match)
- Correct parameter count (256 params for 128 features)
- Proper gradient flow

### 2. Run Training

Execute the main training script:

```bash
python train.py
```

**Training Configuration**:
- Dataset: FashionMNIST
- Epochs: 3
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: AdamW
- Scheduler: OneCycleLR (max_lr=0.01)
- Loss Function: CrossEntropyLoss
- Seed: 42

**Expected Results**:
- Training should complete in approximately 3-5 minutes (CPU) or 1-2 minutes (GPU)
- Test accuracy should reach ~88-90% after 3 epochs
- A `best_model.pth` file will be saved with the best performing model

## Output Files

After training, you will find:
- `best_model.pth` - Saved model weights
- `data/` - Downloaded FashionMNIST dataset (automatically created)

## Hardware Notes

- **CPU**: Training works on CPU but will take longer (~5 minutes for 3 epochs)
- **GPU**: If CUDA is available, training will automatically use GPU and be much faster (~1-2 minutes)

The script automatically detects and uses the appropriate device.

## Verification

To verify your results match the expected output:

1. **Layer Test**: All 5 sanity checks should pass
2. **Training**: Final test accuracy should be within 87-91%
3. **Model Size**: Total parameters should be approximately 1.2M

## Troubleshooting

**Issue**: "RuntimeError: CUDA out of memory"
- **Solution**: Reduce batch size in `train.py` (change `batch_size = 64` to `32` or `16`)

**Issue**: Slow training on Windows
- **Solution**: Set `num_workers=0` in DataLoader (already configured)

**Issue**: Different results despite same seed
- **Solution**: Ensure you're using the same PyTorch version and hardware. Some operations may have slight differences across platforms.

## Version Information

Tested with:
- PyTorch: 2.0.0+
- torchvision: 0.15.0+
- Python: 3.8+

For exact version compatibility, refer to `requirements.txt`.
