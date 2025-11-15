# Diabetes Prediction

## Table of Contents

1. [Problem Description](#problem-description)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [How It Works](#how-it-works)
5. [Installation](#installation)
6. [Running the Project](#running-the-project)
7. [Model Details](#model-details)
8. [Results & Evaluation](#results--evaluation)
9. [Contributing](#contributing)
10. [Contact](#contact)


---

## 1. Problem Description

Diabetes is a serious chronic disease affecting millions worldwide. Predicting who may develop diabetes can help in early detection and prevention, thereby reducing complications and healthcare costs.

In this project 4 machine learning models twere trained to predict whether an individual likely has diabetes based on various health metrics such as **glucose level, BMI, age, blood pressure, insulin**, and the best model was chosen to be deployed. The project demonstrates a complete ML workflow: preprocessing, model training, evaluation, and prediction.

---

## 2. Project Structure

A High-level overview of the repository layout:

```
diabetes-prediction/
├── data/
│   └── diabetes.csv
│
├── notebook.ipynb
│  
│
├── train.py
├── predict.py
│── utils.py
│
├── model_final.bin 
│    
│
├── pyproject.toml  
└── README.md
```



---

## 3. Dataset

This project uses the **Diabetes prediction dataset** that can be found here [data](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

### **Features include:**

* gender
* age
* hypertension
* heart_disease
* smoking_history
* bmi
* HbA1c_level
* blood_glucose_level
* diabetes (target: 1 = diabetes, 0 = no diabetes)

### **Preprocessing performed:**

* Handling of missing values
* Normalization / standardization of features
* Train–test splitting




---

## 4. How It Works

1. **Data Loading & Preprocessing**

   * Missing or zero-values are handled
   * Features are scaled
   * Train–test split is performed
   * Preprocessing utilities live in `train.py` and `utils.py`

2. **Model Training (`train.py`)**

   * Loads the dataset
   * Trains a classification model (Gradient Boosting)
   * Saves the model artifact in `model_final.bin`

3. **Model Saving**

   * Model is saved using `pickle` 
   * Stored inside the main directory

4. **Prediction (`predict.py`)**

   * Loads the saved model
   * Accepts manual input or test samples
   * Produces a diabetes prediction (0 or 1) and optional probability score

---

## 5. Installation

This project uses **UV**, a modern, fast Python package manager that replaces `pip`, `venv`, and `pyenv` with a single tool.
The dependencies are stored in `pyproject.toml`.

### **Prerequisites**

* Python **3.12 or 3.13** (recommended)
* **UV installed**

---

### **Step 1 — Clone the repository**

```bash
git clone https://github.com/Muhammadibra40/diabetes-prediction.git
cd diabetes-prediction
```

---

### **Step 2 — Install UV**

#### **Linux / macOS**

```bash
curl -Ls https://astral.sh/uv/install.sh | bash
```

#### **Windows**

```powershell
iwr https://astral.sh/uv/install.ps1 -useb | iex
```

---

### **Step 3 — Create & activate the environment**

UV automatically manages virtual environments via `uv venv`:

```bash
uv venv
```

Activate it:

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

---

### **Step 4 — Install dependencies from pyproject.toml**

```bash
uv pip install -r pyproject.toml
```

Or simply:

```bash
uv sync
```

(This automatically installs dependencies from your `pyproject.toml`.)

---

## 6. Running the Project

### **Train the model**

```bash
python train.py
```

### **Run predictions**

```bash
python predict.py
```

### **Open the Jupyter Notebook**


run notebook.ipynb


---

## 7. Models Details

* **Algorithms used:** (Random Forest / Logistic Regression / Decision Tree / Gradient Boosting)
* **Input features:** 4 numerical / 4 categorical medical markers
* **Output:** binary prediction (diabetic vs non-diabetic)
* **Chosen model file:** saved at `model_final.bin`


---

## 8. Results & Evaluation of the final chosen model (Gradient Boosting)


## 📊 Model Performance Metrics

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.9078 |
| Precision  | 0.4781 |
| Recall     | 0.9200 |
| F1 Score   | 0.6292 |
| ROC-AUC    | 0.9796 |


---

## 9. Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request


## 10. Contact

**Author:** Muhammad Ibrahim

- **GitHub:** https://github.com/Muhammadibra40  
- **LinkedIn:** https://www.linkedin.com/in/muhammad-ibrahim-093293218/  
- **Email:** migibra678@gmail.com
