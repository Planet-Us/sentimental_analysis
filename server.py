from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Hugging Face 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")
model = AutoModelForSequenceClassification.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    comments = data['comments']

    # 댓글에 대한 감정 분석
    predictions = []
    for comment in comments:
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=512)
        # print(comment)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions.append(prediction.tolist())

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
