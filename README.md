# PyTorch Actor-Critic deep reinforcement learning algorithms: A2C and PPO

The `torch_ac` package contains the PyTorch implementation of two Actor-Critic deep reinforcement learning algorithms:

- [Synchronous A3C (A2C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Proximal Policy Optimizations (PPO)](https://arxiv.org/pdf/1707.06347.pdf)

**Note:** An example of use of this package is given in the [`rl-starter-files` repository](https://github.com/lcswillems/rl-starter-files). More details below.

## Features

- **Recurrent policies**
- Reward shaping
- Wide variety of observation spaces: tensors or dict of tensors
- Wide variety of action spaces: discrete or continuous
- Observation preprocessing
- Multiprocessing
- CUDA

## Installation

```bash
pip3 install torch-ac
```

## Package components overview

A brief overview of the components of the package:

- `torch_ac.A2CAlgo` and `torch_ac.PPOAlgo` classes for A2C and PPO algorithms
- `torch_ac.ACModel` and `torch_ac.RecurrentACModel` abstract classes for non-recurrent and recurrent actor-critic models
- `torch_ac.DictList` class for making dictionnaries of lists list-indexable and hence batch-friendly

## Package components details

Here are detailled the most important components of the package.

`torch_ac.A2CAlgo` and `torch_ac.PPOAlgo` have 2 methods:
- `__init__` that may take, among the other parameters:
    - an `acmodel` actor-critic model, i.e. an instance of a class inheriting from either `torch_ac.ACModel` or `torch_ac.RecurrentACModel`.
    - a `preprocess_obss` function that transforms a list of observations into a list-indexable object `X` (e.g. a PyTorch tensor). The default `preprocess_obss` function converts observations into a PyTorch tensor.
    - a `reshape_reward` function that takes into parameter an observation `obs`, the action `action` taken, the reward `reward` received and the terminal status `done` and returns a new reward. By default, the reward is not reshaped.
    - a `recurrence` number to specify over how many timesteps gradient is backpropagated. This number is only taken into account if a recurrent model is used and **must divide** the `num_frames_per_agent` parameter and, for PPO, the `batch_size` parameter.
- `update_parameters` that first collects experiences, then update the parameters and finally returns logs.

`torch_ac.ACModel` has 2 abstract methods:
- `__init__` that takes into parameter an `observation_space` and an `action_space`.
- `forward` that takes into parameter N preprocessed observations `obs` and returns a PyTorch distribution `dist` and a tensor of values `value`. The tensor of values **must be** of size N, not N x 1.

`torch_ac.RecurrentACModel` has 3 abstract methods:
- `__init__` that takes into parameter the same parameters than `torch_ac.ACModel`.
- `forward` that takes into parameter the same parameters than `torch_ac.ACModel` along with a tensor of N memories `memory` of size N x M where M is the size of a memory. It returns the same thing than `torch_ac.ACModel` plus a tensor of N memories `memory`.
- `memory_size` that returns the size M of a memory.

**Note:** The `preprocess_obss` function must return a list-indexable object (e.g. a PyTorch tensor). If your observations are dictionnaries, your `preprocess_obss` function may first convert a list of dictionnaries into a dictionnary of lists and then make it list-indexable using the `torch_ac.DictList` class as follow:

```python
>>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
>>> d.a
[[1, 2], [3, 4]]
>>> d[0]
DictList({"a": [1, 2], "b": [5]})
```

**Note:** if you use a RNN, you will need to set `batch_first` to `True`.

## Examples

Examples of use of the package components are given in the [`rl-starter-scripts` repository](https://github.com/lcswillems/torch-rl).

An example of use of `torch_ac.A2CAlgo` and `torch_ac.PPOAlgo` classes is given in `scripts/train.py`.

An example of use of `torch_ac.DictList` is given in the `preprocess_obss` functions of `utils/format.py`.

An example of implementation of `torch_ac.RecurrentACModel` abstract class is defined in `model.py`

Examples of `preprocess_obss` functions are given in `utils/format.py`.