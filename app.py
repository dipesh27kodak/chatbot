from flask import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Safe CSV path for Render
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "conv.csv")
data = pd.read_csv(csv_path)

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    chat = ""

    if request.method == "POST":
        old_chat = request.form.get("chat", "")
        qts = request.form["qts"]

        texts = (data["question"].str.lower()).tolist()
        texts.append(qts.lower())

        cv = CountVectorizer()
        vector = cv.fit_transform(texts)

        score = cosine_similarity(vector)
        score = score[-1][:-1]

        data["score"] = score * 100
        result = data.sort_values(by="score", ascending=False)

        if result.iloc[0]["score"] < 10:
            msg = "Bot: Sorry, I don’t know that. Please contact 7498465040."
        else:
            ans = result.head(1)["answer"].values[0]
            msg = "Bot: " + ans

        new_chat = f"You: {qts}\n{msg}\n"
        chat = old_chat + new_chat

        return render_template("home.html", chat=chat.strip())

    return render_template("home.html")


if __name__ == "__main__":
   # app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))