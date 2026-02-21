from flask import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("conv.csv")

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    chat = ""
    if request.method == "POST":
        old_chat = request.form["chat"]
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
            msg = "chitu => sorry i dont know please contact - 7498465040"
        else:
            ans = result.head(1)["answer"].values[0]
            msg = "chitu => " + ans

        new_chat = "you said " + qts + "\n => " + msg
        chat = old_chat + "\n" + new_chat

        return render_template("home.html", msg=msg, chat=chat.strip())

    return render_template("home.html")


#app.run(debug=True, use_reloader=True)