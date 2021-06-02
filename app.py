from flask import Flask,request,render_template
from recommendation_engine import recomendation

app = Flask(__name__,template_folder='template')


@app.route('/')
def hello():
    return """
    <h1>Hello World</h1>
    """

@app.route('/recommend/')
def recommendation_news():
    idx = request.args.get("idx")
    idx = int(idx)
    news_recommend = recomendation(idx)
    keys_recommend = list(news_recommend.keys())
    values_recommend = list(news_recommend.values())
    return render_template("index.html", len = len(keys_recommend), RTitle = keys_recommend, RLink = values_recommend)


if __name__ == '__main__':
    app.run(debug = True, port= 5000)