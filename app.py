import flask
import json
import pickle
from predict import Predcitor

app = flask.Flask(__name__)

predictor = Predcitor("output/model.p")

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this container, we declare
    it healthy if we can load the model successfully."""
    health = (
        predictor.get_predictor_model() is not None
    )

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/classify", methods=["POST"])
def classify():
    data = None
    text = None

    if flask.request.content_type == "application/json":
        print("calling json launched")
        data = flask.request.get_json(silent=True)
        text = data["text"]
    else:
        return flask.Response(
            response="This predictor only supports JSON data",
            status=415,
            mimetype="text/plain",
        )

    if isinstance(text, str):
        prediction = predictor.predict_one(text)
    elif isinstance(text, list):
        prediction = predictor.predict_many(text)
    else:
        return flask.Response(response="Unsuported input text format.", status=415, mimetype="text/plain")
    
    result = json.dumps({"result": prediction})

    return flask.Response(response=result, status=200, mimetype="application/json")
