"""
Build an MLP in JAX from Scratch.

The implementation is deliberately functional: parameters are passed into
functions, random keys are explicit, and optimization returns new parameters
instead of mutating existing ones.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Part 1 - PRNG and random sampling
# ---------------------------------------------------------------------------


def make_prng_key(seed):
    """Wrap an integer seed in a JAX PRNG key."""
    return jax.random.PRNGKey(seed)


def split_prng_key(key, num):
    """Split one PRNG key into ``num`` independent subkeys."""
    return jax.random.split(key, num)


def sample_normal_matrix(key, shape):
    """Sample a standard-normal JAX array with the requested shape."""
    return jax.random.normal(key, shape)


def sample_input_features(key, batch_size, num_features):
    """Generate synthetic input features from a standard normal distribution."""
    return sample_normal_matrix(key, (batch_size, num_features))


def assign_class_labels(inputs, num_classes):
    """Assign each example the class of its largest first ``num_classes`` feature."""
    return jnp.argmax(inputs[:, :num_classes], axis=1).astype(jnp.int32)


def one_hot_encode_labels(labels, num_classes):
    """Convert integer class labels into a one-hot matrix."""
    return jax.nn.one_hot(labels, num_classes)


# ---------------------------------------------------------------------------
# Part 2 - Parameter initialization
# ---------------------------------------------------------------------------


def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Create parameters for one dense layer.

    Returns a dictionary containing W with shape (in_dim, out_dim) and
    b with shape (out_dim,).
    """
    W = sample_normal_matrix(key, (in_dim, out_dim)) * scale
    b = jnp.zeros((out_dim,))
    return {"W": W, "b": b}


def init_mlp_params(key, layer_sizes, scale=0.1):
    """Initialize all dense layers from a sequence of layer sizes."""
    keys = split_prng_key(key, len(layer_sizes) - 1)
    return [
        init_linear_layer(keys[i], layer_sizes[i], layer_sizes[i + 1], scale)
        for i in range(len(layer_sizes) - 1)
    ]


# ---------------------------------------------------------------------------
# Part 3 - Forward pass
# ---------------------------------------------------------------------------


def linear_forward(x, layer_params):
    """Apply a dense layer: x @ W + b."""
    return x @ layer_params["W"] + layer_params["b"]


def relu_activation(x):
    """Apply ReLU elementwise."""
    return jnp.maximum(x, 0)


def softmax_probabilities(logits):
    """Convert logits to numerically stable class probabilities."""
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(shifted)
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)


def mlp_forward(params, x):
    """Run an input batch through all MLP layers."""
    for layer in params[:-1]:
        x = relu_activation(linear_forward(x, layer))
    return linear_forward(x, params[-1])


# ---------------------------------------------------------------------------
# Part 4 - Loss and metrics
# ---------------------------------------------------------------------------


def log_softmax_logits(logits):
    """Compute numerically stable log-softmax values."""
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))


def cross_entropy_loss(logits, one_hot_targets):
    """Return mean multiclass cross-entropy loss."""
    log_probs = log_softmax_logits(logits)
    per_example = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    return jnp.mean(per_example)


def classification_accuracy(logits, labels):
    """Return the fraction of examples classified correctly."""
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


# ---------------------------------------------------------------------------
# Part 5 - Autodiff and SGD
# ---------------------------------------------------------------------------


def loss_fn_of_params(params, x, one_hot_targets):
    """Compute cross-entropy loss for a parameter set and a batch."""
    logits = mlp_forward(params, x)
    return cross_entropy_loss(logits, one_hot_targets)


def compute_param_grads(params, x, one_hot_targets):
    """Differentiate the loss with respect to every parameter tensor."""
    return jax.grad(loss_fn_of_params)(params, x, one_hot_targets)


def sgd_update_params(params, grads, learning_rate):
    """Return parameters after one functional SGD update."""
    return [
        {
            "W": layer["W"] - learning_rate * grad["W"],
            "b": layer["b"] - learning_rate * grad["b"],
        }
        for layer, grad in zip(params, grads)
    ]


def training_step(params, x, one_hot_targets, learning_rate):
    """Perform one gradient-descent step and return (new_params, loss)."""
    loss = loss_fn_of_params(params, x, one_hot_targets)
    grads = jax.grad(loss_fn_of_params)(params, x, one_hot_targets)
    new_params = sgd_update_params(params, grads, learning_rate)
    return new_params, loss


def train_mlp(params, x, one_hot_targets, learning_rate, num_epochs):
    """Train the MLP for ``num_epochs`` full-batch SGD steps."""
    for _ in range(num_epochs):
        params, _ = training_step(params, x, one_hot_targets, learning_rate)
    return params


def predict_classes(params, x):
    """Return predicted class indices for an input batch."""
    logits = mlp_forward(params, x)
    return jnp.argmax(logits, axis=-1)
