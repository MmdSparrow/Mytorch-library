<h1 align="center">Mytorch library</h1>
<h6 align="center">Fall-2024 Computational Intelligence Course Neural Network Project at Amirkabir University of Tech.</h6>

---

# **mytorch**

**mytorch** is a lightweight deep-learning library inspired by PyTorch.
It is designed for learning, experimentation, and building a minimal framework from scratch.
While not intended to replace full-featured frameworks, *mytorch* provides a clean and simple implementation of core neural-network components—ideal for students, researchers, and anyone who wants to understand how deep learning libraries work internally.

---

## Features

* **Tensor Class**
  A simple NumPy-backed tensor implementation with basic mathematical operations.

* **Autograd Engine**
  A minimal automatic-differentiation system that tracks operations and computes gradients through backpropagation.

* **Neural Network Layers**
  Common layers such as:

  * Linear (Fully Connected)
  * Activation functions (ReLU, Sigmoid, Tanh, etc.)

* **Loss Functions**
  Including MSELoss and CrossEntropyLoss.

* **Optimizers**
  Basic optimization algorithms such as SGD and Adam.

* **Model Training Workflow**
  A very small and easy-to-understand training loop for building deep-learning models.

---

## Goals of the Project

* Provide an **educational**, readable implementation of core deep-learning components.
* Help users understand **how PyTorch works internally**, without the complexity.
* Allow small-scale experimentation with autograd and neural networks.

---

## Limitations

Since **mytorch** is intentionally simple, it does **not** include:

* GPU support
* Distributed training

This project is designed for learning and small-scale experiments—not production.

---

## Contributing

Contributions, issue reports, and suggestions are welcome!
Feel free to open a pull request if you want to add features or improve the design.

---
