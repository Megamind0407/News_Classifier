import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Mapping sub-categories to main categories and meaningful names
CATEGORY_MAPPING = {
    'rec.autos': ('Automobile', 'Car Manufacturing & Industry'),
    'rec.motorcycles': ('Automobile', 'Motorcycle Engineering & Riding'),
    'talk.politics.misc': ('Politics', 'General Political Issues'),
    'talk.politics.guns': ('Politics', 'Gun Control & Policy'),
    'talk.politics.mideast': ('Politics', 'Middle East Politics'),
    'sci.space': ('Science', 'Space & Astronomy'),
    'sci.med': ('Science', 'Medical & Health Science'),
    'sci.electronics': ('Science', 'Electronics & Circuits'),
    'sci.crypt': ('Science', 'Cryptography & Security'),
    'comp.graphics': ('Computers', 'Computer Graphics'),
    'comp.os.ms-windows.misc': ('Computers', 'Windows OS Discussions'),
    'comp.sys.ibm.pc.hardware': ('Computers', 'IBM PC Hardware'),
    'comp.sys.mac.hardware': ('Computers', 'Mac Hardware'),
    'comp.windows.x': ('Computers', 'X Windows System')
}

SELECTED_CATEGORIES = list(CATEGORY_MAPPING.keys())

def load_and_prepare_data():
    newsgroups = fetch_20newsgroups(subset='all', categories=SELECTED_CATEGORIES, remove=('headers', 'footers', 'quotes'))
    data = pd.DataFrame({
        'text': newsgroups.data,
        'original_target': [newsgroups.target_names[i] for i in newsgroups.target]
    })
    data['main_category'] = data['original_target'].map(lambda x: CATEGORY_MAPPING[x][0])
    data['sub_category'] = data['original_target'].map(lambda x: CATEGORY_MAPPING[x][1])
    return data

def encode_labels(data):
    main_encoder = LabelEncoder()
    sub_encoder = LabelEncoder()
    data['main_category_encoded'] = main_encoder.fit_transform(data['main_category'])
    data['sub_category_encoded'] = sub_encoder.fit_transform(data['sub_category'])
    return data, main_encoder, sub_encoder

def train_classifier(X, y, model_path):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy and convert to percentage
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f'Accuracy: {accuracy:.2f}%')

    # Save the model
    joblib.dump(pipeline, model_path)
    return pipeline

def main():
    data = load_and_prepare_data()
    data, main_encoder, sub_encoder = encode_labels(data)

    print("Training Main Category Classifier...")
    train_classifier(data['text'], data['main_category_encoded'], 'model/main_category_classifier.pkl')

    print("Training Sub-category Classifier...")
    train_classifier(data['text'], data['sub_category_encoded'], 'model/sub_category_classifier.pkl')

    # Save label encoders
    joblib.dump(main_encoder, 'model/main_category_encoder.pkl')
    joblib.dump(sub_encoder, 'model/sub_category_encoder.pkl')

if __name__ == "__main__":
    main()
