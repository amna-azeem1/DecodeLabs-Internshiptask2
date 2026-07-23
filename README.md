# Iris Flower Species Classification using K-Nearest Neighbors (KNN)

## Overview

This project is a Machine Learning web application that classifies Iris flowers into one of three species: Setosa, Versicolor, and Virginica using the K-Nearest Neighbors (KNN) algorithm.

The application is built with Python, Scikit-learn, and Streamlit, allowing users to enter flower measurements and receive real-time predictions through an interactive web interface.

---

## Features

- Iris flower species classification using KNN
- Interactive Streamlit web application
- Data preprocessing and feature scaling
- Optimal K selection using the Elbow Method
- Model evaluation using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Confidence score for predictions
- Clean and responsive user interface

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- Joblib

---

## Dataset

This project uses the Iris dataset, which contains:

- 150 flower samples
- 4 numerical features
- 3 flower species:
  - Setosa
  - Versicolor
  - Virginica

Features used for prediction:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## Machine Learning Workflow

1. Load and explore the Iris dataset
2. Split the dataset into training and testing sets
3. Apply feature scaling using StandardScaler
4. Train a K-Nearest Neighbors classifier
5. Select the optimal K value using the Elbow Method
6. Evaluate the model using multiple performance metrics
7. Deploy the trained model with Streamlit

---

## Model Performance

- Algorithm: K-Nearest Neighbors (KNN)
- Accuracy: **96.67%**

---

## Project Structure

```
Iris-Flower-Classification/
│
├── app.py
├── iris_classification.py
├── iris_knn_model.pkl
├── scaler.pkl
├── requirements.txt
├── assets/
├── screenshots/
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/DecodeLabs-Internshiptask2.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Improvements

- Support additional classification algorithms
- Improve the user interface
- Deploy the application online
- Add model comparison functionality

---

## Author


**Amna Azeem**

AI & Machine Learning Intern
