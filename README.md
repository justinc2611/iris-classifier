# Iris Classifier (Decision Tree)

## Overview

This project builds a Decision Tree classifier using the Iris dataset from scikit-learn. The model is trained to classify iris flowers into one of three species:

* Setosa
* Versicolor
* Virginica

The project demonstrates a complete machine learning workflow including data loading, train/test splitting, model training, prediction, and evaluation.

## Quick Start

### Clone the repository

```bash
git clone https://github.com/justinc2611/iris-classifier.git
cd iris-classifier
```

### Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the classifier

```bash
python src/train.py --test_size 0.2 --random_state 42
```

## Project Structure

```text
iris-classifier/
├── notebooks/
│   └── iris_model.ipynb
├── src/
│   └── train.py
├── venv
├── LICENSE
├── README.md
└── requirements.txt

## Results

The Decision Tree classifier achieved approximately 100% accuracy using a test size of 0.2 and a random state of 42.

## License

MIT License

