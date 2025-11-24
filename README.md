# Quantization at the Limit: Robustness of Post-Training Quantization in Micro-Scale CNNs

## Overview
This repository contains the code and experiments for the project "Quantization at the Limit: Investigating the Robustness of Post-Training Quantization in Micro-Scale CNNs."

This project reproduces and extends the findings of Jacob et al. (2018) by applying Post-Training Quantization (PTQ) to Convolutional Neural Networks (CNNs) trained on the CIFAR-10 dataset. Specifically, it investigates whether the robustness of 8-bit integer quantization holds for "Micro-Scale" models with extremely limited parameter counts (< 20k parameters).

## Dataset
This project utilizes the **CIFAR-10** and **MNIST** datasets.

* **Source:** Both datasets are loaded directly via the Keras API (`tensorflow.keras.datasets`).
    * `keras.datasets.cifar10.load_data()`
    * `keras.datasets.mnist.load_data()`
* **Description:**
    * **CIFAR-10:** Consists of 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images.
    * **MNIST:** Consists of 60,000 28x28 grayscale images of handwritten digits, used for initial pipeline validation.
* **Note:** No external dataset files need to be downloaded manually; the script will automatically download them to the `~/.keras/datasets/` directory upon the first run.

## Requirements & Dependencies
The code is designed to run in a Python environment (e.g., Google Colab or local Python 3.x).

**Primary Dependencies:**
* **Python 3.x**
* **TensorFlow** (2.x)
* **NumPy**
* **Matplotlib**
* **tensorflow-model-optimization**

*Note: The notebook includes a cell (`!pip install ...`) to automatically install `tensorflow-model-optimization` if it is missing in the environment.*

## How to Run the Experiments

1.  **Open the Notebook:**
    Launch `ECE57000_Final_Project.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.

2.  **Execution Order:**
    Execute the cells in sequential order. The notebook is structured as follows:
    * **Setup:** Imports libraries and sets random seeds (Seed=42) for reproducibility.
    * **MNIST Experiment (Validation):**
        * Builds and trains a simple CNN on MNIST.
        * Evaluates the FP32 baseline.
        * Quantizes the model to INT8 using TFLite.
        * Evaluates the INT8 model to verify the quantization pipeline.
    * **CIFAR-10 Standard Experiment:**
        * Trains a standard CNN (~350k params) on CIFAR-10.
        * Converts to INT8 and compares performance.
    * **CIFAR-10 Micro Experiment (Main Contribution):**
        * Defines a custom "Micro" CNN architecture (~10k params).
        * Trains the Micro model on CIFAR-10.
        * Quantizes to INT8 and evaluates robustness.
    * **Analysis & Visualization:**
        * Generates the Pareto Efficiency chart comparing Accuracy vs. Latency for all models.

3.  **Outputs:**
    * The notebook will print accuracy, latency, and model size metrics for both FP32 and INT8 versions of all models.
    * A final Pareto chart (`pareto_chart.png`) will be generated within the notebook.

## Key Results
The experiments demonstrate that Post-Training Quantization is robust even for micro-scale models.
* **Micro-Model Accuracy Drop:** < 0.1% (Negligible)
* **Inference Speedup:** ~208x (on TFLite interpreter vs. Keras execution)
* **Model Compression:** ~10x reduction in size (165KB -> 16.7KB)

## License
MIT License
