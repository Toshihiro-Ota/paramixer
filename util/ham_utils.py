import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import equinox as eqx



# for step 1: normalization
from einops import rearrange

def transform(x):
    x = x / 255.
    x = rearrange(x, "... h w -> ... (h w)")
    x = x / jnp.sqrt((x ** 2).sum(-1, keepdims=True))
    return x



# for step 2:
import optax

def lossf(
        ham,
        xs,
        key,
        nsteps=1,  # number of steps to update
        alpha=1.,  # step size to take in direction of updates
        ):
    """Given a noisy initial image, descend the energy and try to reconstruct the original image at the end of the dynamics.

    Works best with fewer steps due to vanishing gradient problems"""
    input = xs['input']
    xs['input'] = input + jr.normal(key, input.shape) * 0.3

    for _ in range(nsteps):
        # Construct noisy image to final prediction
        gs = ham.activations(xs)
        evalue, egrad = ham.dEdg(gs, xs, return_energy=True)
        xs = jtu.tree_map(lambda x, dEdg: x - alpha * dEdg, xs, egrad)  # The negative of our dEdg, computing the update direction each layer should descend

    gs = ham.activations(xs)
    output = gs['input']
    loss = ((output - input)**2).mean()

    logs = {
        "loss": loss,
    }
    return loss, logs

@eqx.filter_jit
def step(
    input,
    ham,
    opt_state,
    key,
    opt=None,  # optax.adam(lr)
    nsteps=1,  # number of steps to update
    alpha=1.,  # step size to take in direction of updates
    ):
    xs = ham.init_states(bs=input.shape[0])
    xs['input'] = input

    (loss, logs), grads = eqx.filter_value_and_grad(lossf, has_aux=True)(ham, xs, key, nsteps=nsteps, alpha=alpha)
    updates, opt_state = opt.update(grads, opt_state, ham)
    newparams = optax.apply_updates(eqx.filter(ham, eqx.is_array), updates)
    ham = eqx.combine(newparams, ham)
    return ham, opt_state, logs



# for step 3: set the colormap and center the colorbar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    """Normalise the colorbar."""
    def __init__(
            self,
            vmin=None,
            vmax=None,
            midpoint=None,
            clip=False,
            ):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def show_img(img, figsize=(6, 6)):
    vmin, vmax = img.min(), img.max()
    vscale = max(np.abs(vmin), np.abs(vmax))
    cnorm = MidpointNormalize(midpoint=0., vmin=-vscale, vmax=vscale)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    pcm = ax.imshow(img, cmap="seismic", norm=cnorm)
    ax.axis("off")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    fig.colorbar(pcm, cax=cbar_ax);
    return fig
