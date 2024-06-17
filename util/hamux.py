"""HAMUX, a minimal implementation of the Hierarchical Associative Memory

HAMUX is the skeleton of what could be an entirely new way to build DL architectures using energy blocks.
"""
# source: https://github.com/bhoov/barebones-hamux/blob/main/bbhamux.py

from typing import Union, Callable, Tuple, Dict, List, Optional
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jr
import equinox as eqx


class Neurons(eqx.Module):
    """Neurons represent dynamical variables in HAM that are evolved during inference (i.e., memory retrieval / error correction)

    They have an evolving state (created using the `.init` function) that is stored outside the neuron layer itself
    """

    lagrangian: Union[Callable, eqx.Module]
    shape: Tuple[int]

    def __init__(
            self,
            lagrangian: Union[Callable, eqx.Module],
            shape: Union[int, Tuple[int]],
            ):
        self.lagrangian = lagrangian
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape

    def __repr__(self: jax.Array):
        return f"Neurons(lagrangian={self.lagrangian}, shape={self.shape})"

    def init(self, bs: Optional[int] = None) -> jax.Array:  # bs: batch size
        """Initialize the states of this layer, with correct shape.

        If `bs` is provided, return tensor of shape (bs, *self.shape), otherwise return self.shape
        By default, initialize layer state to all 0.
        """
        if bs is None or bs == 0:
            return jnp.zeros(self.shape)
        return jnp.zeros((bs, *self.shape))

    def activations(self, x: jax.Array) -> jax.Array:
        """Compute the activations of the neuron layer, the derivative of the Lagrangian"""
        return jax.grad(self.lagrangian)(x)

    def g(self, x: jax.Array) -> jax.Array:
        """Alias for the `activations`"""
        return self.activations(x)

    def energy(
            self,
            g: jax.Array,
            x: jax.Array,
            ) -> jax.Array:
        """Compute the energy. Assume neurons are vectorized"""
        return jnp.multiply(g, x).sum() - self.lagrangian(x)


#* Example Lagrangians
"""Default Lagrangian functions that correspond to commonly used non-linearities in neural networks.

1. Lagrangians return a scalar
2. Lagrangians are convex
3. The derivative of a Lagrangian w.r.t. its input is the activation function typically used in neural networks

"""

def lagr_identity(x):
    """The Lagrangian whose activation function is simply the identity."""
    return 0.5 * jnp.power(x, 2).sum()

def lagr_repu(x, n):
    """Rectified Power Unit of degree `n`"""  # n: degree of the polynomial in the power unit
    return 1 / n * jnp.power(jnp.maximum(x, 0), n).sum()

def lagr_relu(x):
    """Rectified Linear Unit. Same as `repu` of degree 2"""
    return lagr_repu(x, 2)

def lagr_softmax(
        x: jnp.ndarray,
        beta: float = 1.0,  # inverse temperature
        axis: int = -1,  # axis over which to apply logsumexp
        ):
    """The Lagrangian of the softmax -- the logsumexp"""
    return 1 / beta * jax.nn.logsumexp(beta * x, axis=axis, keepdims=False)

def lagr_layernorm(
        x: jnp.ndarray,
        gamma: float = 1.0,  # scale the std
        delta: Union[float, jnp.ndarray] = 0.0,  # shift the mean
        axis: int = -1,  # which axis to normalize
        eps: float = 1e-5,  # prevent division by 0
        ):
    """Lagrangian of the layer norm activation function"""

    D = x.shape[axis] if axis is not None else x.size
    xmean = x.mean(axis, keepdims=True)
    xmeaned = x - xmean
    y = jnp.sqrt(jnp.power(xmeaned, 2).mean(axis, keepdims=True) + eps)

    return (D * gamma * y + (delta * x).sum()).sum()

def lagr_spherical_norm(
        x: jnp.ndarray,
        gamma: float = 1.0,  # scale the std
        delta: Union[float, jnp.ndarray] = 0.0,  # shift the mean
        axis: int = -1,  # which axis to normalize
        eps: float = 1e-5,  # prevent division by 0
        ):
    """Lagrangian of the spherical norm activation function"""

    y = jnp.sqrt(jnp.power(x, 2).sum(axis, keepdims=True) + eps)

    return (gamma * y + (delta * x).sum()).sum()


#* Example Synapses.
class DenseSynapse(eqx.Module):
    """The simplest of dense (linear) functions that defines the energy between two layers"""

    W: jax.Array

    def __init__(
            self,
            key: jax.Array,
            g1_dim: int,
            g2_dim: int,
            ):
        super().__init__()
        # simplest initialization
        self.W = 0.02 * jr.normal(key, (g1_dim, g2_dim)) + 0.2

    @property
    def nW(self):
        nc = jnp.sqrt(jnp.sum(self.W ** 2, axis=0, keepdims=True))
        return self.W / nc

    def __call__(
            self,
            g1: jax.Array,
            g2: jax.Array,
            ):
        """Compute the energy between activations g1 and g2.

        The more aligned, the lower the energy"""
        return -jnp.einsum("...c,...d,cd->...", g1, g2, self.W)

class DenseSynapseHid(eqx.Module):
    W: jax.Array
    def __init__(
            self,
            key: jax.Array,
            d1: int,
            d2: int,
            ):
        super().__init__()
        self.W = 0.02 * jr.normal(key, (d1, d2)) + 0.2

    @property
    def nW(self):
        nc = jnp.sqrt(jnp.sum(self.W ** 2, axis=0, keepdims=True))
        return self.W / nc

    def __call__(self, g1: jax.Array):
        """Compute the energy of the synapse.

        Here logsumexp lagrangian is assumed"""
        x2 = g1 @ self.nW
        beta = 1e1
        return - 1/beta * jax.nn.logsumexp(beta * x2, axis=-1)



class HAM(eqx.Module):
    """The Hierarchical Associative Memory

    A wrapper for all dynamic states (neurons) and learnable parameters (synapses) of our memory
    """

    neurons: Dict[str, Neurons]
    synapses: Dict[str, eqx.Module]
    connections: List[Tuple[Tuple, str]]

    def __init__(
            self,
            neurons: Dict[str, Neurons],  # Neurons are the dynamical variables expressing the state of the HAM
            synapses: Dict[str, eqx.Module],  # Synapses are the learnable relationships between dynamic variables.
            connections: List[Tuple[Tuple[str, ...], str]],  # Connections expressed as [(['ni', 'nj'], 'sk'), ...]. Read as "Connect neurons 'ni' and 'nj' via synapse 'sk'
            ):
        """An HAM is a hypergraph that connects neurons and synapses together via connections"""
        self.neurons = neurons
        self.synapses = synapses
        self.connections = connections

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_synapses(self) -> int:
        return len(self.synapses)

    @property
    def n_connections(self) -> int:
        return len(self.connections)

    def init_states(self, bs: Optional[int] = None):  # If provided, each neuron in the HAM has this batch size
        """Initialize neuron states"""
        xs = {k: v.init(bs) for k, v in self.neurons.items()}
        return xs

    def activations(
            self,
            xs: Dict[str, jax.Array],  # The expected collection of neurons states
            ) -> Dict[str, jax.Array]:
        """Convert hidden states of each neuron into activations"""
        gs = {k: v.g(xs[k]) for k, v in self.neurons.items()}
        return gs

    def neuron_energies(
            self,
            gs: Dict[str, jax.Array],
            xs: Dict[str, jax.Array],
            ):
        """Return the energies of each neuron in the HAM"""
        return {k: self.neurons[k].energy(gs[k], xs[k]) for k in self.neurons.keys()}

    def connection_energies(
        self,
        gs: Dict[str, jax.Array],  # The collection of neuron activations
    ):
        """Get the energy for each connection

        A function of the activations `gs` rather than the states `xs`
        """

        def get_energy(neuron_set, s):
            temp = [gs[k] for k in neuron_set]
            return self.synapses[s](*temp)

        return [get_energy(neuron_set, s) for neuron_set, s in self.connections]

    def energy_tree(self, gs, xs):
        """Return energies for each individual component"""
        neuron_energies = self.neuron_energies(gs, xs)
        connection_energies = self.connection_energies(gs)
        return {"neurons": neuron_energies, "connections": connection_energies}

    def energy(self, gs, xs):
        """The complete energy of the HAM"""
        energy_tree = self.energy_tree(gs, xs)
        return jtu.tree_reduce(lambda E, acc: acc + E, energy_tree, 0)

    def dEdg(
            self,
            gs,
            xs,
            return_energy=False,
            ):
        """Calculate the gradient of system energy w.r.t. activations using cute trick:

        The derivative of the neuron energy w.r.t. the activations is the neuron state itself.
        This is a property of the Legendre transform:
        dE_layer / dg = x
        """

        def all_connection_energy(gs):
            return jtu.tree_reduce(lambda E, acc: acc + E, self.connection_energies(gs), 0)

        dEdg = jtu.tree_map(lambda x, s: x + s, xs, jax.grad(all_connection_energy)(gs))  # xs = dE_layers / dgs
        if return_energy:
            return self.energy(gs, xs), dEdg
        return jax.grad(self.energy)(gs, xs)

    def vectorize(self):
        """Compute a new HAM with same API, all methods expect batch dimension"""
        return VectorizedHAM(self)

    def unvectorize(self):
        return self


class VectorizedHAM(eqx.Module):
    """Re-expose HAM API with vectorized inputs. No logic should be implemented in this class."""

    _ham: eqx.Module

    def __init__(self, ham):
        self._ham = ham

    @property
    def neurons(self):
        return self._ham.neurons

    @property
    def synapses(self):
        return self._ham.synapses

    @property
    def connections(self):
        return self._ham.connections

    @property
    def n_neurons(self):
        return self._ham.n_neurons

    @property
    def n_synapses(self):
        return self._ham.n_synapses

    @property
    def n_connections(self):
        return self._ham.n_connections

    @property
    def _batch_axes(self: HAM):
        """A helper function to tell vmap to batch along the 0th dimension of each state in the HAM."""
        return {k: 0 for k in self._ham.neurons.keys()}

    def init_states(self, bs=None):
        """Initialize neuron states with batch size `bs`"""
        return self._ham.init_states(bs)

    def activations(self, xs):
        """Compute activations of a batch of inputs"""
        return jax.vmap(self._ham.activations, in_axes=(self._batch_axes,))(xs)

    def neuron_energies(self, gs, xs):
        """Compute energy of every neuron in the HAM"""
        return jax.vmap(self._ham.neuron_energies, in_axes=(self._batch_axes, self._batch_axes))(gs, xs)

    def connection_energies(self, gs):
        """Compute energy of every connection in the HAM"""
        return jax.vmap(self._ham.connection_energies, in_axes=(self._batch_axes,))(gs)

    def energy_tree(self, gs, xs):
        """Return energies for each individual component"""
        return jax.vmap(self._ham.energy_tree, in_axes=(self._batch_axes, self._batch_axes))(gs, xs)

    def energy(self, gs, xs):
        """Compute the energy of the entire HAM"""
        return jax.vmap(self._ham.energy, in_axes=(self._batch_axes, self._batch_axes))(gs, xs)

    def dEdg(self, gs, xs, return_energy=False):
        """Compute the gradient of the energy wrt the activations of the HAM"""
        return jax.vmap(self._ham.dEdg, in_axes=(self._batch_axes, self._batch_axes, None))(gs, xs, return_energy)

    def unvectorize(self):
        """Return an HAM energy that does not work on batches of inputs"""
        return self._ham

    def vectorize(self):
        return self
