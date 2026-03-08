# Hybrid Deep Learning Models for Potato Plant Disease Detection and Classification

This project implements and evaluates deep learning models for potato plant disease detection and identification using image datasets. It provides a framework for training and testing various Convolutional Neural Network (CNN), Graph Neural Network (GNN) and Vision Transformer (ViT) based architectures, including hybrid models.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Models](#models)
- [Logging](#logging)

## Project Overview

This project builds upon the methodology described in [Sinamenye, J.H. et al, 2025], which implemented potato plant disease detection using EfficientNetV2B3. In this repository, we extend the original work by experimenting with **`hybrid models`**, including **`EffNetGNN`**, which were implemented from scratch. These modifications explore hybrid architectures for improved disease detection performance while keeping the original dataset and task consistent.

EffNetGNN is a hybrid deep learning architecture that combines EfficientNet’s powerful feature extraction capabilities with Graph Neural Networks (GNNs) to model spatial relationships and contextual dependencies between image regions through learnable graph structures. By leveraging these hybrid features, EffNetGNN achieves state-of-the-art performance in potato leaf disease classification.

The repository also provides a framework for training and testing various CNN, GNN, and Vision Transformer (ViT) based architectures, including other hybrid models, for comparative analysis.

## Project Structure

Plant-Disease-Detection/
│
├── src/
│ ├── train.py # Training script
│ ├── test.py # Evaluation/testing script
│ ├── models.py # Model architectures (EffNetGNN, EffNetViT, etc.)
│ └── configurations.py # Training/testing and dataset configuration
│
├── README.md # Project documentation
├── pyproject.toml  
├── poetry.lock  
├── requirements.txt # Optional dependencies
├── .gitignore # Git ignore file
└── .pre-commit-config.yaml # Pre-commit hooks configuration

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <https://github.com/recadet419/Plant-Disease-Detection.git>
    cd <Plant-Disease-Detection>
    ```

2.  **Ensure Poetry is installed:**
    If you don't have Poetry installed, follow the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

3.  **Install dependencies using Poetry:**
    Navigate to the project root (where `pyproject.toml` is located) and run:

    ```bash
    poetry install
    ```

    This will create a virtual environment if one doesn't exist and install all the dependencies specified in `pyproject.toml`.

4.  **Activate the virtual environment (if needed):**
    Poetry often manages virtual environments automatically. To run commands within the project's environment, you can use `poetry run <your_command>` (e.g., `poetry run python src/train.py`).
    Alternatively, you can activate the environment directly using `poetry shell`.

5.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your Weights & Biases API key and project details:
    ```
    WANDB_KEY="your_wandb_api_key"
    WANDB_PROJECT="your_wandb_project_name"
    WANDB_ENTITY="your_wandb_entity"
    ```

## Dataset

The project used the publicly available dataset, Potato Leaf Disease Dataset available [here] (https://data.mendeley.com/datasets/ptz377bwb8/1)

## Configuration

All major settings for training, testing, model selection, and dataset paths are managed in `src/configurations.py`. Key configurations include:

- `DATA`: Path to the dataset.
- `TEST_SIZE`, `VALI_SIZE`: Ratios for splitting the dataset.
- `BATCH_SIZE`: Batch size for training and evaluation.
- `CLASSES`: Automatically determined from the dataset directory.
- `DEVICE`: Automatically detects CUDA availability.
- `CRITERION`: Loss function (e.g., `nn.CrossEntropyLoss()`).
- `EPOCHS`: Number of training epochs.
- `lr`: Learning rate.
- `TRAINING`: Boolean flag to switch between training (`True`) and testing (`False`) modes.
- `AUGMENT`: Boolean flag to enable/disable data augmentation.
- `DATATYPE`: Name of the dataset being used (e.g., "potatodata").
- `NEW_DATASET`: Flag for specific saved model paths, often used to differentiate between experiments.
- `MODELS`: Dictionary defining the models to be used.
- `OPTIMIZERS`: Dictionary defining optimizers for each model.
- `SCHEDULER`: Dictionary defining learning rate schedulers.
- `SAVED_MODELS`: Dictionary mapping model names to their saved checkpoint file paths, conditional on `NEW_DATASET` and `AUGMENT` flags.

Modify this file to experiment with different settings.

## Usage

Ensure you have configured `src/configurations.py` appropriately before running the scripts.

### Training

To train a model:

1.  Set `TRAINING = True` in `src/configurations.py`.
2.  Configure other relevant parameters like `EPOCHS`, `lr`, `MODEL` to train, `DATATYPE`, etc.
3.  Run the training script:
    ```bash
    python src/train.py
    ```
    Trained models will be saved, and progress will be logged to Weights & Biases.

### Testing

To test a trained model:

1.  Set `TRAINING = False` in `src/configurations.py`.
2.  Ensure the `SAVED_MODELS` dictionary in `src/configurations.py` points to the correct trained model checkpoint for the `EfficientNetGNN` (often keyed as `EffNetGNN`) or other model you wish to test.
3.  Specify the model to test in the `MODELS` dictionary (it should reference the class, not an instance).
4.  Run the testing script:
    ```bash
    python src/test.py
    ```
    Evaluation metrics will be printed, and results can also be logged to Weights & Biases.

## Models

The foundation of this project is the **`EffNetGNN`** hybrid model which combines the strengths of EfficientNet (a Convolutional Neural Network (CNN)) and Graph Neural Networks (GNN). This model, defined in `src/models.py` (and typically referenced as `EfficientNetGNN` in `src/configurations.py`), is the primary subject of the research publication due to its promising results in potato plant disease detection and identification.

While `EffNetGNN` is the main focus, the framework also includes implementations of other models for comparative analysis, such as:

- `EffNetViT`
- `ConvNeXtGNN`
- `ConvNeXtViT`
- `ConvNeXt`

These can be configured and experimented with via `src/configurations.py`.

## Logging

Experiment tracking and results are logged using [Weights & Biases](https://wandb.ai). Ensure your `.env` file is configured with your W&B credentials. The run name for logging is dynamically generated in `src/configurations.py` based on the timestamp, dataset type, and augmentation status.

---

_This README provides a general guide. Modify paths and commands as per your specific setup._
