from flask import Flask, request, render_template
import pickle

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
app = Flask(__name__)  

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        return render_template("index.html", result=sentiment)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
