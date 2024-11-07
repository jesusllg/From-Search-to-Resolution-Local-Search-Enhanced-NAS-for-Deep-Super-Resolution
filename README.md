# Neural Architecture Search with Local Search and NSGA-III Integration

Welcome to the **Neural Architecture Search (NAS) with Local Search and NSGA-III Integration** repository. This project implements various local search algorithms and integrates them with the NSGA-III evolutionary algorithm for multi-objective optimization in neural architecture search.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Performance Comparison](#performance-comparison)
- [How It Works](#how-it-works)
  - [Encoding and Decoding](#encoding-and-decoding)
  - [Model Building](#model-building)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Optimization Process](#optimization-process)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running the Algorithms](#running-the-algorithms)
- [Algorithms Implemented](#algorithms-implemented)
  - [Local Search Methods](#local-search-methods)
    - [Hill Climbing](#hill-climbing)
    - [Tabu Search](#tabu-search)
    - [Simulated Annealing](#simulated-annealing)
  - [NSGA-III](#nsga-iii)
  - [NSGA-III with Local Search Integration](#nsga-iii-with-local-search-integration)
- [Research Publications](#research-publications)
- [Contributing](#contributing)
- [License](#license)
- [Quick Links](#quick-links)

---

## Introduction

This repository focuses on optimizing neural network architectures using multi-objective optimization techniques. The primary goal is to find optimal architectures that balance performance metrics like PSNR (Peak Signal-to-Noise Ratio), model complexity (number of parameters), and computational cost (FLOPs).

## Features

- **Local Search Algorithms**: Implementation of Hill Climbing, Tabu Search, and Simulated Annealing tailored for NAS.
- **NSGA-III Algorithm**: Integration of the NSGA-III evolutionary algorithm for multi-objective optimization.
- **Hybrid Approach**: A novel algorithm combining NSGA-III with periodic local search refinements.
- **Customizable Evaluation Metrics**: Supports PSNR and SynFlow as evaluation metrics.
- **Modular Design**: Organized code structure for easy understanding and modification.
- **Extensive Documentation**: Code is thoroughly commented, and the README provides a comprehensive guide.

## Performance Comparison

The following table compares the performance of various algorithms based on SynFlow, number of Parameters, and FLOPs.

| **Algorithm**          | **SynFlow Average**    | **SynFlow Std. Dev.**  | **Parameters Average**   | **Parameters Std. Dev.** | **FLOPs Average**        | **FLOPs Std. Dev.**      |
|------------------------|------------------------|------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| NSGA-III<sup>*</sup>   | $6.16 \times 10^{4}$   | $7.63 \times 10^{3}$   | $2.39 \times 10^{5}$     | $6.29 \times 10^{5}$     | $9.82 \times 10^{8}$     | $2.58 \times 10^{9}$     |
| NSGA-III-HC<sup>*</sup>| **$6.45 \times 10^{4}$** | $3.39 \times 10^{3}$ | $2.97 \times 10^{5}$     | $6.19 \times 10^{4}$     | **$2.23 \times 10^{8}$**| $2.54 \times 10^{8}$     |
| NSGA-III-TS<sup>*</sup>| $6.36 \times 10^{4}$   | $4.95 \times 10^{3}$   | **$8.18 \times 10^{4}$**| $4.72 \times 10^{4}$     | $4.36 \times 10^{8}$     | $1.26 \times 10^{9}$     |
| NSGA-III-SA<sup>*</sup>| $5.80 \times 10^{4}$   | $7.73 \times 10^{3}$   | $2.36 \times 10^{5}$     | $4.24 \times 10^{5}$     | $5.36 \times 10^{8}$     | $1.74 \times 10^{9}$     |

<sup>*</sup> Indicates that the algorithm integrates local search methods.

**Notes:**

- **Bold Values**: Highlighted values indicate the best performance in their respective categories.
- **SynFlow**: A proxy metric used for evaluating neural network architectures without full training.
- **Parameters**: Number of parameters in the neural network model.
- **FLOPs**: Floating Point Operations, indicating the computational cost.

## How It Works

### Encoding and Decoding

- **Genome Representation**: Neural network architectures are represented as bitstrings (genomes).
- **Encoding**: The `encoding.py` module handles the conversion of bitstrings to genotypes, which are structured representations of the architectures.
- **Decoding**: The genotypes are decoded to build actual Keras models using `model_builder.py`.

### Model Building

- **Branching Structure**: Models are built with three branches, each defined by the genotype.
- **Layer Primitives**: Various convolutional and identity operations are used, defined in the `PRIMITIVES` list.
- **Dynamic Construction**: The `get_model` function constructs the model dynamically based on the genotype.

### Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Used to measure the quality of reconstructed images in tasks like super-resolution.
- **SynFlow**: A proxy metric used for evaluating neural network architectures without training them fully.
- **FLOPs and Parameters**: Computational cost is measured using FLOPs, and model complexity is assessed by counting the number of parameters.

### Optimization Process

- **Local Search**: Starts with random architectures and refines them using bit-flip mutations, guided by Pareto optimality.
- **NSGA-III**: Evolves a population of architectures over multiple generations, promoting diversity and convergence to the Pareto front.
- **Hybrid Approach**: Integrates local search into NSGA-III, refining individuals periodically to enhance search efficiency.

## Project Structure

```bash
your-repo-name/
├── config.py
├── encoding.py
├── evaluation.py
├── local_search.py
├── main.py
├── model_builder.py
├── nsga3.py
├── utils.py
├── README.md
├── LICENSE       # If you have a license file
└── requirements.txt  # List of dependencies
```

- **config.py**: Configuration settings for the project.
- **encoding.py**: Encoding and decoding of genome representations.
- **evaluation.py**: Evaluation functions for PSNR, SynFlow, FLOPs, and parameter counting.
- **local_search.py**: Implementation of Hill Climbing, Tabu Search, and Simulated Annealing.
- **main.py**: Main script to run the optimization process.
- **model_builder.py**: Functions to build Keras models from genotypes.
- **nsga3.py**: Implementation of NSGA-III and NSGA-III with Local Search.
- **utils.py**: Utility functions, including dominance checks and PSNR calculation.
- **README.md**: This document.
- **LICENSE**: License information.
- **requirements.txt**: Python dependencies.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd your-repo-name
   ```

3. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have TensorFlow, NumPy, and other necessary libraries installed.

## Usage

### Configuration

Before running the algorithms, you need to configure the settings in `config.py`:

- **Evaluation Metric**: Set `EVALUATION_METRIC` to `'PSNR'` or `'SynFlow'`.
- **Datasets**: Replace the placeholders in `DATASET_TRAIN` and `DATASET_VAL` with your actual training and validation datasets.
- **Local Search Config**: Adjust parameters like `MAX_EVALUATIONS`, `TABU_TENURE`, `INITIAL_TEMP`, and `COOLING_RATE`.
- **Device Configuration**: Specify whether to use GPU or CPU.

```python
# config.py

import numpy as np
import random
import tensorflow as tf

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Evaluation metric: 'PSNR' or 'SynFlow'
EVALUATION_METRIC = 'PSNR'  # You can change to 'SynFlow' as needed

# Local Search Configuration
LOCAL_SEARCH_CONFIG = {
    'MAX_EVALUATIONS': 25000,      # Total evaluations per local search
    'TABU_TENURE': 5,              # Tenure for Tabu Search
    'INITIAL_TEMP': 100,           # Initial temperature for Simulated Annealing
    'COOLING_RATE': 0.95,          # Cooling rate for Simulated Annealing
}

# Dataset Configuration
# Replace with your actual datasets
DATASET_TRAIN = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
DATASET_VAL = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)
EPOCHS = 5  # Number of epochs for training when using PSNR

# Device Configuration
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
```

### Running the Algorithms

Use `main.py` to execute the optimization process. You can select the algorithm to run by setting the `algorithm_choice` variable.

```python
# In main.py
algorithm_choice = 'NSGA-III with Local Search'  # Options: 'HillClimbing', 'TabuSearch', 'SimulatedAnnealing', 'NSGA-III', 'NSGA-III with Local Search'
```

Run the script:

```bash
python main.py
```

## Algorithms Implemented

### Local Search Methods

#### Hill Climbing

An iterative algorithm that starts with an arbitrary solution and makes incremental changes to the solution, each time improving it based on the objective function.

#### Tabu Search

An advanced local search method that uses memory structures called tabu lists to avoid cycles and encourage exploration of new areas in the solution space.

#### Simulated Annealing

A probabilistic technique that explores the solution space by allowing occasional uphill moves, thus avoiding local minima and potentially finding a global minimum.

### NSGA-III

An evolutionary algorithm designed for solving complex multi-objective optimization problems. It uses reference points to maintain diversity and spread among the solutions.

### NSGA-III with Local Search Integration

A hybrid algorithm that combines NSGA-III with periodic local search refinements. Every 50 generations, a random individual from the population is selected and refined using a specified local search method for 1,000 evaluations.

## Research Publications

This project is part of ongoing research in neural architecture search and multi-objective optimization. The following articles have been published as a result of this work:

1. **Title of Your Article 1**: *Journal/Conference Name*, Year. [Link](https://example.com/article1)
2. **Title of Your Article 2**: *Journal/Conference Name*, Year. [Link](https://example.com/article2)
3. **Title of Your Article 3**: *Journal/Conference Name*, Year. [Link](https://example.com/article3)

*(Note: Replace the placeholders with actual article titles and links.)*

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**.
2. **Create a new branch** for your feature or bug fix.
3. **Commit your changes** with clear messages.
4. **Push to your fork**.
5. **Submit a pull request**.

Please ensure your code adheres to the project's coding standards and includes proper documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*For any questions or issues, please open an issue on GitHub or contact the maintainers.*

## Quick Links

- [Project Repository](https://github.com/yourusername/your-repo-name)
- [Issues](https://github.com/yourusername/your-repo-name/issues)
- [Pull Requests](https://github.com/yourusername/your-repo-name/pulls)

---

Thank you for your interest in this project!
