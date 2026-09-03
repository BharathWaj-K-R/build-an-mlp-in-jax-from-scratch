import numpy as np
import jax.numpy as jnp

from model import (
    make_prng_key,
    split_prng_key,
    sample_normal_matrix,
    sample_input_features,
    assign_class_labels,
    one_hot_encode_labels,
    init_linear_layer,
    init_mlp_params,
    linear_forward,
    relu_activation,
    softmax_probabilities,
    mlp_forward,
    log_softmax_logits,
    cross_entropy_loss,
    classification_accuracy,
    loss_fn_of_params,
    compute_param_grads,
    sgd_update_params,
    training_step,
    train_mlp,
    predict_classes,
)


def test_prng_and_sampling_shapes():
    key = make_prng_key(0)
    keys = split_prng_key(key, 3)
    assert keys.shape == (3, 2)
    matrix = sample_normal_matrix(keys[0], (4, 5))
    assert matrix.shape == (4, 5)
    x = sample_input_features(keys[1], 7, 5)
    assert x.shape == (7, 5)


def test_labels_and_one_hot():
    x = jnp.array([[1.0, 2.0, 0.0], [3.0, -1.0, 4.0]])
    labels = assign_class_labels(x, 3)
    np.testing.assert_array_equal(labels, jnp.array([1, 2]))
    y = one_hot_encode_labels(labels, 3)
    np.testing.assert_array_equal(y, jnp.array([[0, 1, 0], [0, 0, 1]]))


def test_layer_and_mlp_shapes():
    layer = init_linear_layer(make_prng_key(1), 3, 4)
    assert layer["W"].shape == (3, 4)
    assert layer["b"].shape == (4,)
    params = init_mlp_params(make_prng_key(2), [3, 5, 4], scale=0.1)
    assert len(params) == 2
    out = linear_forward(jnp.ones((2, 3)), params[0])
    assert out.shape == (2, 5)


def test_activations_and_probabilities():
    x = jnp.array([[-1.0, 0.0, 2.0]])
    np.testing.assert_array_equal(relu_activation(x), jnp.array([[0.0, 0.0, 2.0]]))
    probs = softmax_probabilities(x)
    np.testing.assert_allclose(jnp.sum(probs, axis=1), 1.0)
    np.testing.assert_allclose(jnp.exp(log_softmax_logits(x)), probs)


def test_loss_accuracy_and_gradients():
    params = init_mlp_params(make_prng_key(3), [3, 5, 2], scale=0.1)
    x = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    labels = jnp.array([0, 1])
    y = one_hot_encode_labels(labels, 2)
    logits = mlp_forward(params, x)
    loss = cross_entropy_loss(logits, y)
    accuracy = classification_accuracy(logits, labels)
    assert np.isfinite(float(loss))
    assert 0.0 <= float(accuracy) <= 1.0
    grads = compute_param_grads(params, x, y)
    assert len(grads) == len(params)
    assert grads[0]["W"].shape == params[0]["W"].shape


def test_training_improves_loss_on_easy_problem():
    x = jnp.array([
        [3.0, 0.0, -1.0],
        [2.5, 0.0, -0.5],
        [0.0, 3.0, -1.0],
        [0.0, 2.5, -0.5],
        [-1.0, 0.0, 3.0],
        [-0.5, 0.0, 2.5],
    ])
    labels = assign_class_labels(x, 3)
    y = one_hot_encode_labels(labels, 3)
    params = init_mlp_params(make_prng_key(4), [3, 8, 8, 3], scale=0.1)
    initial_loss = float(loss_fn_of_params(params, x, y))
    params_after, step_loss = training_step(params, x, y, 0.1)
    assert np.isfinite(float(step_loss))
    trained = train_mlp(params_after, x, y, 0.1, 300)
    final_loss = float(loss_fn_of_params(trained, x, y))
    predictions = predict_classes(trained, x)
    assert final_loss < initial_loss
    assert float(classification_accuracy(mlp_forward(trained, x), labels)) >= 0.99
    np.testing.assert_array_equal(predictions, labels)


def test_sgd_does_not_mutate_original_parameters():
    params = init_mlp_params(make_prng_key(5), [2, 3, 2])
    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    y = one_hot_encode_labels(jnp.array([0, 1]), 2)
    grads = compute_param_grads(params, x, y)
    updated = sgd_update_params(params, grads, 0.1)
    assert any(not np.array_equal(np.asarray(a["W"]), np.asarray(b["W"])) for a, b in zip(params, updated))
