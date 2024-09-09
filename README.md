# Cucumber Harvest Prediction Using Neural Networks

This project demonstrates a neural network approach to predict the optimal harvesting time for cucumbers based on environmental factors. The dataset used in this project is **randomly generated** and does not contain real data. It has been created solely for demonstration purposes and to showcase machine learning skills in a GitHub portfolio.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Skills Learned](#skills-learned)
- [Acknowledgments](#acknowledgments)

## Project Overview
The goal of this project is to develop a neural network model capable of predicting cucumber weight, which helps determine the best time for harvesting. The model takes two inputs:
- **Hour**: Hour of the day (1-24).
- **PAR (Photosynthetically Active Radiation)**: A light intensity measure relevant to plant growth.

This project uses a simple neural network and examines its performance in predicting cucumber weights using the generated data.

## Dataset
- The dataset used in this project is **randomly generated** and is not representative of actual field data.
- The purpose of this dataset is to showcase the implementation of neural networks without violating any data privacy or NDA agreements.

## Model Architecture
The neural network model used in this project includes:
- Input layer specifying the shape of the data.
- Multiple Dense layers with ReLU activation functions.
- Dropout layers to prevent overfitting.
- Output layer with linear activation for regression output.

## Dependencies
To run this project, ensure you have the following dependencies installed:
- Python 3.9+
- TensorFlow
- scikit-learn
- numpy
- pandas
- matplotlib

You can install the necessary dependencies using pip:

  ```bash
  pip install tensorflow scikit-learn numpy pandas matplotlib
  ```
## Usage
1. Clone this repository:
 
  ```bash
  git clone https://github.com/your-username/cucumber-harvest-prediction.git
  ```
2. Navigate to the project directory:
 
  ```bash
  cd cucumber-harvest-prediction
  ```
3. Open the Jupyter Notebook:

  ```bash
  jupyter notebook CucumberPredictions.ipynb
  ```
4. Run the cells sequentially to train the model and visualize the results.

## Results
The model demonstrates the capability to fit the synthetic data, providing predictions on cucumber weight based on input features. Due to the nature of the generated dataset, these results are intended for demonstration purposes only.

## Skills Learned
- Developed skills in building and training neural networks using TensorFlow.
- Gained experience in handling synthetic datasets and validating model performance.
- Enhanced understanding of data preprocessing, feature scaling, and model evaluation.

## Acknowledgments
This project uses synthetic data for demonstration purposes and does not reflect actual agricultural data.
The TensorFlow library was used for building and training the neural network.

## Disclaimer
This project is for educational purposes only. The dataset used is artificially generated and not intended for real-world application.
