# Build an MLP in JAX from Scratch

Implement a multi-layer perceptron end-to-end in JAX, from PRNG key handling and synthetic data generation to parameter initialization, forward propagation, stable cross-entropy loss, automatic differentiation, SGD training, and inference.

The implementation is intentionally functional: JAX PRNG keys are explicit, parameters are immutable, and optimization returns new parameter structures instead of mutating existing ones.

## How to run

```bash
python scaffold.py
```

Requirements:

```bash
pip install "jax[cpu]" numpy
```

## Implementation checklist

- [x] **1.** `make_prng_key`
- [x] **2.** `split_prng_key`
- [x] **3.** `sample_normal_matrix`
- [x] **4.** `sample_input_features`
- [x] **5.** `assign_class_labels`
- [x] **6.** `one_hot_encode_labels`
- [x] **7.** `init_linear_layer`
- [x] **8.** `init_mlp_params`
- [x] **9.** `linear_forward`
- [x] **10.** `relu_activation`
- [x] **11.** `softmax_probabilities`
- [x] **12.** `mlp_forward`
- [x] **13.** `log_softmax_logits`
- [x] **14.** `cross_entropy_loss`
- [x] **15.** `classification_accuracy`
- [x] **16.** `loss_fn_of_params`
- [x] **17.** `compute_param_grads`
- [x] **18.** `sgd_update_params`
- [x] **19.** `training_step`
- [x] **20.** `train_mlp`
- [x] **21.** `predict_classes`

## Project structure

```text
.
├── model.py       # Complete MLP implementation
├── scaffold.py    # End-to-end executable demonstration
├── README.md      # Project documentation
└── docs/
    └── index.html # Project walkthrough
```

## What the implementation does

1. Creates explicit JAX PRNG keys.
2. Generates a synthetic classification dataset.
3. Initializes a configurable MLP from a list of layer sizes.
4. Performs dense layers, ReLU activation, and logits computation.
5. Computes numerically stable log-softmax and cross-entropy loss.
6. Uses `jax.grad` to obtain gradients for every layer.
7. Applies functional full-batch SGD updates.
8. Trains for a configurable number of epochs.
9. Produces class predictions and reports loss/accuracy before and after training.

## Notes

The synthetic labels are deliberately constructed from the largest value among the first `num_classes` input features. This creates a simple deterministic learning problem that can be used to verify the complete training pipeline without downloading an external dataset.

Built on Deep-ML.
