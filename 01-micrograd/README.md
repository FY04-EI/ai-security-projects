Part of my AI security roadmap — see [my profile](https://github.com/FY04-EI).

# micrograd — a scalar autograd engine from scratch

A reimplementation of a reverse-mode automatic differentiation engine and a small multilayer perceptron, written from scratch in pure Python.

## Contents

- `engine.py` — the `Value` class: computation graph and backpropagation
- `nn.py` — `Neuron`, `Layer` and `MLP`, built on `Value`
- `demo.ipynb` — gradient checks and a training run with its loss curve

The engine has no external dependency. Matplotlib is only used by the demo.

## Run it

    uv sync
    uv run jupyter lab
