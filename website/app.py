from flask import Flask, render_template, request, jsonify
import openai
import xgboost as xgb
import numpy as np
import pickle

app = Flask(__name__)

# OpenAI API 키 설정
openai.api_key = "YOUR_OPENAI_API_KEY"

# XGBoost 모델 불러오기
with open("xgboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# 📌 1. 조회수 예측 함수
def predict_views(title, thumbnail, hashtags, description):
    # 임시: 입력값을 숫자로 변환하는 간단한 예제 (학습 데이터에 맞게 수정 필요)
    features = np.array([len(title), len(thumbnail), len(hashtags), len(description)]).reshape(1, -1)
    
    predicted_views = model.predict(features)
    return int(predicted_views[0])  # 정수형 조회수 반환

# 📌 2. YouTube 제목 추천 함수
def generate_youtube_titles(user_input):
    prompt = f"Generate 5 engaging YouTube titles for a video about '{user_input}'"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"].split("\n")

# 📌 3. 메인 페이지 (메뉴 선택)
@app.route("/")
def home():
    return render_template("index.html")

# 📌 4. 조회수 예측 페이지
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        title = request.form["title"]
        thumbnail = request.form["thumbnail"]
        hashtags = request.form["hashtags"]
        description = request.form["description"]

        predicted_views = predict_views(title, thumbnail, hashtags, description)
        return jsonify({"views": predicted_views})

    return render_template("predict.html")

# 📌 5. YouTube 제목 추천 페이지
@app.route("/generate_title", methods=["GET", "POST"])
def generate_title():
    if request.method == "POST":
        user_input = request.form["title"]
        titles = generate_youtube_titles(user_input)
        return jsonify({"titles": titles})

    return render_template("title_generate.html")

if __name__ == "__main__":
    app.run(debug=True)
