"""
Custom PyTorch layers for FashionMNIST classification.
"""
import torch
import torch.nn as nn


class LearnedAffine(nn.Module):
    """
    A custom affine transformation layer that learns scale and shift parameters.
    
    This layer applies: y = scale * x + shift
    where scale and shift are learnable parameters initialized to 1 and 0 respectively.
    
    Args:
        num_features: Number of features in the input
    """
    def __init__(self, num_features):
        super(LearnedAffine, self).__init__()
        self.num_features = num_features
        
        # Initialize scale to 1 and shift to 0
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        """
        Apply affine transformation.
        
        Args:
            x: Input tensor of shape (..., num_features)
        
        Returns:
            Transformed tensor of same shape as input
        """
        return self.scale * x + self.shift
    
    def extra_repr(self):
        """String representation for printing the layer."""
        return f'num_features={self.num_features}'


def test_learned_affine():
    """
    Sanity check for LearnedAffine layer.
    Tests shape preservation and parameter count.
    """
    print("=" * 60)
    print("LearnedAffine Sanity Check")
    print("=" * 60)
    
    # Test 1: Shape preservation
    batch_size = 32
    num_features = 128
    
    layer = LearnedAffine(num_features)
    x = torch.randn(batch_size, num_features)
    y = layer(x)
    
    print(f"\nTest 1: Shape Preservation")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert x.shape == y.shape, "Output shape should match input shape"
    print("  ✓ PASSED")
    
    # Test 2: Parameter count
    expected_params = 2 * num_features  # scale + shift
    actual_params = sum(p.numel() for p in layer.parameters())
    
    print(f"\nTest 2: Parameter Count")
    print(f"  Expected: {expected_params}")
    print(f"  Actual:   {actual_params}")
    assert actual_params == expected_params, f"Expected {expected_params} params, got {actual_params}"
    print("  ✓ PASSED")
    
    # Test 3: Parameter names and shapes
    print(f"\nTest 3: Parameter Details")
    for name, param in layer.named_parameters():
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    # Test 4: Initial values
    print(f"\nTest 4: Initial Values")
    print(f"  Scale initialized to 1s: {torch.allclose(layer.scale, torch.ones(num_features))}")
    print(f"  Shift initialized to 0s: {torch.allclose(layer.shift, torch.zeros(num_features))}")
    
    # Test 5: Gradient flow
    print(f"\nTest 5: Gradient Flow")
    x.requires_grad = True
    y = layer(x)
    loss = y.sum()
    loss.backward()
    print(f"  Scale gradient exists: {layer.scale.grad is not None}")
    print(f"  Shift gradient exists: {layer.shift.grad is not None}")
    print(f"  Input gradient exists: {x.grad is not None}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_learned_affine()
