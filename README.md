PyTorch Linear Regression Tutorial
This repository contains a Jupyter Notebook (PyTorch2.ipynb) that demonstrates an end-to-end workflow for building and training a simple linear regression model using PyTorch.

Table of Contents
Introduction
Dependencies
Data Preparation
Data Visualization
Model Building
Training the Model
Model Evaluation
Saving and Loading the Model
Introduction
This notebook walks through the process of creating a linear regression model in PyTorch, including data preparation, model building, training, evaluation, and saving/loading the model. The original notebook is hosted on Google Colab and can be accessed here.

Dependencies
Ensure you have the following libraries installed before running the notebook:

torch
matplotlib
You can install these dependencies using pip:

bash
Copy code
pip install torch matplotlib
Data Preparation
We prepare the data for our linear regression model using the equation 
𝑦
=
𝑎
+
𝑏
𝑋
y=a+bX with the following parameters:

weight = 0.7
bias = 0.3
The data is split into training (80%) and testing (20%) sets.

Data Visualization
A function plot_predictions is defined to visualize the training data, test data, and model predictions.

Model Building
We define a simple linear regression model class LinearRegressionModel using PyTorch's nn.Module. The model uses the formula 
𝑦
=
weights
∗
𝑥
+
bias
y=weights∗x+bias.

Training the Model
The training loop involves:

Forward pass
Loss calculation using nn.L1Loss
Backpropagation
Optimizer step using stochastic gradient descent (SGD)
Model Evaluation
The model is evaluated on the test set, and predictions are plotted to visualize the model's performance.

Saving and Loading the Model
The trained model's state dictionary is saved to a file and can be loaded later for inference or further training.

Saving the Model
python
Copy code
from pathlib import Path

# Create model's directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path
MODEL_NAME = "01_linearRegModel.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model's state dict
torch.save(obj=model0.state_dict(), f=MODEL_SAVE_PATH)
Loading the Model
python
Copy code
# Instantiate a new model instance
load_model0 = LinearRegressionModel()

# Load the saved state dictionary
load_model0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
Usage
To use this notebook, open it in Jupyter or Google Colab and run the cells sequentially. The notebook includes code cells for each step in the workflow, from data preparation to model evaluation and saving/loading the model.

Conclusion
This notebook provides a comprehensive guide to building and training a simple linear regression model using PyTorch. It covers essential concepts such as data preparation, model building, training loops, and saving/loading models, providing a solid foundation for further exploration and experimentation with PyTorch.
