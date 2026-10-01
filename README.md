# FY04-EI

Background in networks and cybersecurity: firewalling, segmentation, Linux systems.
I am currently building machine learning from the ground up, with a focus on how these systems can be attacked and defended.
I study with Claude as a tutor: it asks questions and challenges my reasoning rather than giving answers. I write the code myself (except the plotting code).

## Projects

- **[micrograd](https://github.com/FY04-EI/ai-security-projects/tree/main/01-micrograd)**: a scalar autograd engine and a small neural network, re-implemented from scratch following Andrej Karpathy's *Neural Networks: Zero to Hero* (episode 1). *In progress.*
  I implemented backpropagation by hand: recording the computation graph, ordering it topologically, and accumulating gradients through the chain rule, then trained a small MLP with gradient descent.

## Currently working on

- **Image classifier**: small CNNs trained from scratch on CIFAR-10, evaluated on the held-out test set, with reproducible runs.
- **Text classifier**: sentiment analysis on movie reviews.

Both will be published in the same repository once complete.
