from flask import Flask, request, jsonify, render_template
from transformers import BartForConditionalGeneration, BartTokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved BART model and tokenizer
model_name = "bart_model"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, max_length=130, min_length=30, length_penalty=2.0, num_beams=4):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    summary = summarize_text(text)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
