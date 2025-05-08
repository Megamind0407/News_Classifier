from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load models and encoders
main_model = joblib.load('model/main_category_classifier.pkl')
sub_model = joblib.load('model/sub_category_classifier.pkl')
main_encoder = joblib.load('model/main_category_encoder.pkl')
sub_encoder = joblib.load('model/sub_category_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['news_text']

        # Predict main and sub-category
        main_pred = main_model.predict([text])[0]
        sub_pred = sub_model.predict([text])[0]

        # Decode predictions
        main_category = main_encoder.inverse_transform([main_pred])[0]
        sub_category = sub_encoder.inverse_transform([sub_pred])[0]

        return render_template('result.html', 
                               text=text,
                               main_category=main_category,
                               sub_category=sub_category)

if __name__ == '__main__':
    app.run(debug=True)
