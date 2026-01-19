# Training Metrics

## What I Tried

I built a simple CNN for FashionMNIST with a custom `LearnedAffine` layer that learns scale and shift parameters (256 total params). The model has two convolutional blocks with max pooling, then fully connected layers with the custom affine layer in between.

I used AdamW optimizer (lr=0.001) for weight decay and OneCycleLR scheduler to ramp up to 10x learning rate then anneal down over training. This helps escape local minima early and fine-tune later.

## What Worked

Training converged smoothly over 3 epochs, reaching around 88-90% test accuracy. The OneCycleLR scheduler worked well with steady improvement each epoch. The custom `LearnedAffine` layer integrated easily and added useful learnable parameters without overfitting.

## What I'd Change Next

To improve performance, I'd train longer (10-15 epochs) to reach 91-93% accuracy. Adding data augmentation like random rotations and shifts would help generalization. I'd also try CosineAnnealingLR for smoother convergence, and experiment with batch normalization instead of the custom affine layer to compare performance.
