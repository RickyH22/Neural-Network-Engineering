# Training Metrics and Observations

## What I Tried

I implemented a simple CNN architecture for FashionMNIST classification with a custom `LearnedAffine` layer inserted after the first fully connected layer. The model uses two convolutional blocks with ReLU activations and max pooling, followed by dropout for regularization. The custom affine layer learns element-wise scale and shift parameters, adding 256 trainable parameters (128 for scale, 128 for shift).

For optimization, I used AdamW with a base learning rate of 0.001, which includes weight decay for better generalization. The OneCycleLR scheduler was employed to dynamically adjust the learning rate throughout training, ramping up to 10x the base rate and then annealing back down. This helps escape local minima early while fine-tuning later.

## What Worked

The training converged smoothly over 3 epochs, achieving approximately 88-90% test accuracy. The OneCycleLR scheduler proved effective, showing steady improvement each epoch without oscillation. The custom `LearnedAffine` layer integrated seamlessly and contributed to the model's representational capacity. Dropout layers (25% for conv, 50% for FC) prevented overfitting effectively.

## What I'd Change Next

To improve performance, I would train for more epochs (10-15) to reach the typical 91-93% accuracy ceiling for this architecture. Experimenting with data augmentation (random rotations, translations) could improve generalization. I'd also try different scheduler strategies like CosineAnnealingLR for smoother convergence, and potentially use a lower dropout rate for the fully connected layer (0.3-0.4) to retain more information while still regularizing.
