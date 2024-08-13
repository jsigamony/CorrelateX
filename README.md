# 📊 PyTorch Linear Regression Tutorial

This repository contains a Jupyter Notebook (`PyTorch2.ipynb`) demonstrating an end-to-end workflow for building and training a simple linear regression model using PyTorch.

## 📑 Table of Contents

1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Data Visualization](#data-visualization)
6. [Model Building](#model-building)
7. [Training the Model](#training-the-model)
8. [Model Evaluation](#model-evaluation)
9. [Saving and Loading the Model](#saving-and-loading-the-model)
10. [Usage](#usage)

## 🎉 Introduction

This notebook walks through the process of creating a linear regression model in PyTorch, including data preparation, model building, training, evaluation, and saving/loading the model. The original notebook is hosted on [Google Colab](https://colab.research.google.com/drive/your-notebook-link-here).

## 🛠️ Dependencies

- Python 3.7+
- PyTorch
- Matplotlib

## 💻 Installation

Install the required dependencies using pip:

```bash
pip install torch matplotlib
🔢 Data Preparation
We prepare the data for our linear regression model using the equation y = a + bX with:

weight (b) = 0.7
bias (a) = 0.3
The data is split into training (80%) and testing (20%) sets.

📈 Data Visualization
A plot_predictions function is defined to visualize the training data, test data, and model predictions.

🏗️ Model Building
We define a LinearRegressionModel class using PyTorch's nn.Module. The model uses the formula y = weights * x + bias.

🏋️‍♀️ Training the Model
The training loop involves:

Forward pass
Loss calculation using nn.L1Loss
Backpropagation
Optimizer step using stochastic gradient descent (SGD)
📊 Model Evaluation
The model is evaluated on the test set, and predictions are plotted to visualize performance.

💾 Saving and Loading the Model
Saving the Model
python

from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_linearRegModel.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model0.state_dict(), f=MODEL_SAVE_PATH)
Loading the Model
python

 
load_model0 = LinearRegressionModel()
load_model0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
🚀 Usage
To use this notebook:

Open it in Jupyter or Google Colab
Run the cells sequentially
Experiment with different parameters or datasets
