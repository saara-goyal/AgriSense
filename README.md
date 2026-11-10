# AgriSense 🌱
### An AI-Powered Decision Support System for Smart Agriculture

AgriSense is a machine-learning and deep-learning-based web application designed to support agricultural decision-making through crop recommendations, fertilizer suggestions, and plant disease detection.

## 🌾 Features

### 1. Crop Recommendation
- Accepts soil and environmental parameters:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - Soil pH
  - Rainfall
- Uses a trained Random Forest model to recommend a suitable crop.

### 2. Fertilizer Recommendation
- Accepts crop name and soil NPK values.
- Analyses nutrient differences.
- Provides fertilizer suggestions based on the available dataset.

### 3. Plant Disease Detection
- Allows users to upload a plant leaf image.
- Uses a trained ResNet9 deep-learning model.
- Predicts the corresponding plant disease or healthy condition.

## 🛠️ Technologies Used

- Python
- Flask
- NumPy and Pandas
- Scikit-learn
- PyTorch and Torchvision
- HTML, CSS, JavaScript and Bootstrap

## 📁 Project Structure

```text
AgriSense/
│
├── app.py
├── config.py
├── requirements.txt
├── Data/
│   └── fertilizer.csv
├── models/
│   ├── RandomForest.pkl
│   ├── plant_disease_model.zip
│   ├── DecisionTree.pkl
│   └── NBClassifier.pkl
├── templates/
├── static/
└── utils/
```

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/saara-goyal/AgriSense.git
cd AgriSense
```

Alternatively, download the repository as a ZIP and extract it.

### 2. Set up the disease detection model

Open the `models` folder and extract:

`plant_disease_model.zip`

Make sure the extracted file is named:

`plant_disease_model.pth`

It must be located directly inside the `models` folder.

### 3. Set up the Python environment

Create and activate a Python virtual environment:

```bash
python -m venv agri_env
