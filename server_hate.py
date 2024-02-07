from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify
from torch.nn.functional import sigmoid
import torch

app = Flask(__name__)

# Hugging Face 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("beomi/korean-hatespeech-multilabel")
model = AutoModelForSequenceClassification.from_pretrained("beomi/korean-hatespeech-multilabel")

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    comments = data['comments']
    # print(comments)

    # 댓글에 대한 감정 분석
    predictions = []
    for comment in comments:
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = sigmoid(outputs.logits)  # Softmax 대신 Sigmoid 사용
            predictions.append(prediction.tolist())

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
