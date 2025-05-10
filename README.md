# 📰 News Classifier

A machine learning project to classify news articles into **main categories** and **sub-categories** using a **Random Forest Classifier**, trained on the [20 Newsgroups dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html). A Flask API is integrated for real-time predictions.

---

## 🚀 Features

- Classifies news into high-level categories like **Politics**, **Science**, **Computers**, **Automobile**
- Further predicts detailed sub-categories (e.g., *Gun Control*, *Cryptography*, *Space & Astronomy*)
- Utilizes TF-IDF Vectorization and Random Forest for robust text classification
- Built-in Flask API to serve predictions
- Supports model saving and reusability with `joblib`

---
![image](https://github.com/user-attachments/assets/5b289e5a-aeda-4510-a91c-fbc6d95b52b8)

![image](https://github.com/user-attachments/assets/75f2b954-c235-4203-9210-f9772a7040af)

## 🧠 Model Overview

- **Dataset**: Subset of the 20 Newsgroups dataset
- **Preprocessing**:
  - Removed headers, footers, and quotes
  - TF-IDF vectorization for feature extraction
- **Model**: Random Forest Classifier
- **Encoders**: LabelEncoders for main and sub-category labels
- **Accuracy**: ~90%+ (depends on run)

---

## 🗃️ Category Mapping

Each raw label is mapped to a human-friendly **Main Category** and **Sub-category**.  
Example:

| Original Label          | Main Category | Sub-category                      |
|------------------------|---------------|------------------------------------|
| rec.autos              | Automobile    | Car Manufacturing & Industry       |
| sci.crypt              | Science       | Cryptography & Security            |
| talk.politics.mideast  | Politics      | Middle East Politics               |

---

## 🛠️ Project Structure
```
news-classifier/
├── model/ # Saved model & encoders
│ ├── main_category_classifier.pkl
│ ├── sub_category_classifier.pkl
│ ├── main_category_encoder.pkl
│ └── sub_category_encoder.pkl
├── app.py # Flask API
├── train.py # Training pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/news-classifier.git
cd news-classifier
```
### 2.Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Train the models
```bash
python train.py
```
This will train and save two classifiers:
```
main_category_classifier.pkl

sub_category_classifier.pkl
```
### 4. Run the Flask API
```bash
python app.py
```
API will be live at: `http://127.0.0.1:5000`

## 📩 Example API Usage
### POST /predict
Request:
```
{
  "text": "NASA launched a new satellite to monitor climate change."
}
```
Response:
```
{
  "main_category": "Science",
  "sub_category": "Space & Astronomy"
}
```
## 🧪 Dependencies
### numpy
### pandas
### scikit-learn
### joblib
### flask

To install all dependencies:
```bash
pip install numpy pandas scikit-learn joblib flask
```
## 🙌 Acknowledgements
### Scikit-learn
### 20 Newsgroups Dataset
### Flask for API deployment

