import jax
import numpy as np
from PIL import Image
import jax.numpy as jnp
import optax
from functools import partial

_SIZE = 128
_ITERATIONS = 1000
_GS_NUM = 5000


@partial(jax.vmap, in_axes=(None, None, None, 0))  # map over height dimension
@partial(jax.vmap, in_axes=(None, None, None, 0))  # map over width dimension
@partial(jax.vmap, in_axes=(0, 0, 0, None))  # map over gaussians
def compute_gaussians(mu, theta, scaling, coord):
    """Compute the value of a single Gaussian at the given coordinate."""
    # rotation matrix
    # clip theta to the range [0, pi]
    theta = jnp.clip(theta, 0.0, jnp.pi)
    c = jnp.cos(theta[0])
    s = jnp.sin(theta[0])
    R = jnp.array([[c, -s], [s, c]])
    # scaling matrix
    # clip scaling to be positive
    scaling = jnp.clip(scaling, min=1e-6, max=None)
    S = jnp.diag(scaling)
    # covariance matrix
    # add small value for numerical stability
    Sigma = R @ S @ S @ R.T + 1e-6 * jnp.eye(2)
    diff = coord - mu
    exponent = -0.5 * diff @ jnp.linalg.inv(Sigma) @ diff.T
    return jnp.exp(exponent)


@jax.jit
def render_image(mu_array, theta_array, scaling_array, color_array, coords):
    """Render the image from the parameters."""
    gaussians = compute_gaussians(mu_array, theta_array, scaling_array, coords)
    # weighted average of gaussians
    rendered_image = jnp.matmul(gaussians, color_array) / (
        jnp.sum(gaussians, axis=-1, keepdims=True) + 1e-6
    )
    rendered_image = jnp.clip(rendered_image, 0, 255)
    return rendered_image


def main():
    img = Image.open("input.jpg")
    img_array = np.array(img)
    key = jax.random.PRNGKey(0)
    key_mu, key_theta, key_scaling, key_color = jax.random.split(key, 4)
    mu_array = jax.random.uniform(key_mu, (_GS_NUM, 2), minval=0.0, maxval=1.0)
    theta_array = jax.random.uniform(key_theta, (_GS_NUM, 1), minval=0.0, maxval=jnp.pi)
    scaling_array = jax.random.uniform(
        key_scaling, (_GS_NUM, 2), minval=0.0, maxval=0.1
    )
    color_array = jax.random.uniform(key_color, (_GS_NUM, 3), minval=0.0, maxval=255.0)

    # create a grid of x, y coordinates
    x = jnp.linspace(0, 1.0, _SIZE)
    y = jnp.linspace(0, 1.0, _SIZE)
    xx, yy = jnp.meshgrid(x, y)
    coords = jnp.stack([xx, yy], axis=-1)  # shape (_SIZE, _SIZE, 2)

    learning_rate = 0.001
    optimizer = optax.adam(learning_rate)
    params = (mu_array, theta_array, scaling_array, color_array)
    opt_state = optimizer.init(params)

    def loss_fn(params):
        rendered = render_image(
            *params,
            coords,
        )
        return jnp.mean(jnp.abs(rendered - img_array))

    def update(params, opt_state):
        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    for i in range(_ITERATIONS):
        if i % 10 == 0:
            r_image = render_image(
                *params,
                coords,
            )
            img = Image.fromarray(np.array(r_image.astype(jnp.uint8)))
            img.save(f"output_{i:04d}.png")

        params, opt_state = update(params, opt_state)
        current_loss = loss_fn(params)
        print(f"Step: {i}, Loss: {current_loss}")


if __name__ == "__main__":
    main()
