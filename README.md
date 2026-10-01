# FY04-EI

Background in networks and cybersecurity: firewalling, segmentation, Linux systems.
I am currently building machine learning from the ground up, with a focus on how these systems can be attacked and defended.

## Projects

- **[micrograd](https://github.com/FY04-EI/ai-security-projects/tree/main/01-micrograd)**: a scalar autograd engine and a small neural network, written from scratch. *In progress.*
  I implemented backpropagation by hand: recording the computation graph, ordering it topologically, and accumulating gradients through the chain rule, then trained a small MLP with gradient descent. The same gradient, taken with respect to the input instead of the weights, is what adversarial attacks exploit.

## Currently working on

- **Image classifier**: small CNNs trained from scratch on CIFAR-10, evaluated on the held-out test set, with reproducible runs.
- **Text classifier**: sentiment analysis on movie reviews.

Both will be published in the same repository once complete.
