from flask import Flask, request, render_template, jsonify
import re
import nltk
from nltk.corpus import stopwords
import pickle

nltk.download('stopwords')

app = Flask(__name__)

# Load the vectorizer and models
with open('vectorizer.pkl', 'rb') as file:
    vectorization = pickle.load(file)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocess_text(text_data): 
    preprocessed_text = [] 
    for sentence in text_data: 
        sentence = re.sub(r'[^\w\s]', '', sentence) 
        preprocessed_text.append(' '.join(token.lower() 
                                          for token in str(sentence).split() 
                                          if token not in stopwords.words('english'))) 
    return preprocessed_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['news_text']
    preprocessed_input = preprocess_text([input_text])
    input_vector = vectorization.transform(preprocessed_input)
    prediction = model.predict(input_vector)
    credibility_score = model.predict_proba(input_vector)[0][1] * 100  # Confidence score for class 1
    return jsonify(prediction=int(prediction[0]), credibility_score=credibility_score)

if __name__ == '__main__':
    app.run(debug=True)
