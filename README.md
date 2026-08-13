# 🫀 Heart Disease Analyzer

An interactive **Machine Learning-based Heart Disease Prediction Web Application** built using **Python and Streamlit**. The application takes clinical health parameters as input and uses a trained **K-Nearest Neighbors (KNN)** model to predict heart disease risk.

## 🚀 Project Overview

Heart Disease Analyzer is a machine learning project designed to demonstrate how clinical health data can be processed and used for classification.

Users can enter various health and cardiac parameters through an interactive web interface. The application preprocesses the input data, applies the saved scaler, and sends the processed data to the trained KNN model to generate a prediction.

The application provides:

* 🫀 Heart disease risk prediction
* 📊 Model risk probability when available
* ⚡ Instant prediction results
* 📋 Input information review
* 🎨 Modern healthcare-themed interface
* 🔄 Reset functionality
* ⚠️ Medical disclaimer

---

## 🧠 Machine Learning Model

The project uses a **K-Nearest Neighbors (KNN) Classifier**.

The trained model is loaded from:

```text
knn_heart_model.pkl
```

The application also loads:

```text
heart_scaler.pkl
heart_columns.pkl
```

The scaler is used to transform the input data before it is passed to the model, while `heart_columns.pkl` is used to maintain the feature-column structure expected by the model.

### Prediction

The model produces a binary prediction:

```text
0 → Lower Risk
1 → Higher Risk
```

If the trained model supports `predict_proba()`, the application also displays the model's predicted probability.

---

## 📊 Input Features

The application collects the following parameters:

| Feature         | Description                           |
| --------------- | ------------------------------------- |
| Age             | Patient's age                         |
| Sex             | Patient's sex                         |
| Chest Pain Type | Type of chest pain                    |
| Resting BP      | Resting blood pressure                |
| Cholesterol     | Cholesterol level                     |
| Fasting BS      | Fasting blood sugar above 120 mg/dL   |
| Resting ECG     | Resting electrocardiogram result      |
| Max HR          | Maximum heart rate                    |
| Exercise Angina | Exercise-induced angina               |
| Oldpeak         | ST depression value                   |
| ST Slope        | Slope of the peak exercise ST segment |

---

## ⚙️ How It Works

The application follows this workflow:

```text
User Input
    ↓
Create Input DataFrame
    ↓
Match Expected Model Columns
    ↓
Feature Scaling
    ↓
KNN Model
    ↓
Prediction
    ↓
Risk Result
```

### 1. User Input

The user enters their health parameters through the Streamlit interface.

### 2. Data Preparation

The entered values are converted into a Pandas DataFrame.

Categorical variables are converted into the encoded feature format expected by the trained model.

### 3. Feature Alignment

The application checks the columns stored in `heart_columns.pkl`.

Missing columns are automatically added with a value of `0`, and the columns are reordered to match the model's expected structure.

### 4. Feature Scaling

The input is transformed using the saved scaler:

```python
scaled_input = scaler.transform(input_df)
```

### 5. Prediction

The scaled input is passed to the trained KNN model:

```python
prediction = model.predict(scaled_input)[0]
```

### 6. Result

The application displays either:

* 💚 **Lower Risk Detected**
* ⚠️ **Higher Risk Detected**

---

## 🎨 Application Interface

The Streamlit application includes several sections:

### 👤 Personal Information

Collects:

* Age
* Sex
* Resting Blood Pressure

### ❤️ Heart & Blood Parameters

Collects:

* Cholesterol
* Fasting Blood Sugar
* Maximum Heart Rate

### 🩺 ECG & Exercise Information

Collects:

* Chest Pain Type
* Resting ECG
* Exercise-Induced Angina

### 📈 ST Segment Parameters

Collects:

* Oldpeak
* ST Slope

### 📋 Prediction Result

After clicking **Analyze Heart Health**, the application displays the model's prediction and, when supported, the model risk probability.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* K-Nearest Neighbors (KNN)

### Data Processing

* Pandas

### Model Serialization

* Joblib

### Web Framework

* Streamlit

### UI

* Streamlit
* HTML
* CSS

---

## 📁 Project Structure

```text
HEART-DISEASE-ANALYZER/
│
├── app.py
├── knn_heart_model.pkl
├── heart_scaler.pkl
├── heart_columns.pkl
├── requirements.txt
└── README.md
```

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shami0002/HEART-DISEASE-ANALYZER.git
```

### 2. Navigate to the project directory

```bash
cd HEART-DISEASE-ANALYZER
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will then open in your browser.

---

## 📦 Requirements

The application requires the following main Python packages:

```text
streamlit
pandas
scikit-learn
joblib
```

You can create the `requirements.txt` file with:

```bash
pip freeze > requirements.txt
```

---

## 🔮 Future Improvements

Some possible improvements for future versions include:

* 📊 Add model accuracy and other evaluation metrics
* 📈 Add confusion matrix and ROC-AUC visualization
* 🤖 Compare KNN with other classification algorithms
* 📉 Add feature importance and model interpretation
* 📄 Generate downloadable prediction reports
* 🌐 Deploy the application online
* 📱 Improve mobile responsiveness
* 🔐 Add secure user authentication

---

## ⚠️ Medical Disclaimer

This application is an **educational machine learning project**.

The predictions generated by this application should **not be considered a medical diagnosis** and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

If you have concerns about your heart health or are experiencing serious symptoms, consult a qualified healthcare professional.

---

## 👨‍💻 Author

**Shami**

Machine Learning Project

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

**Repository:**
`https://github.com/Shami0002/HEART-DISEASE-ANALYZER`
