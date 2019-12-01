import pickle

class Predcitor(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None 
        self.get_predictor_model()

    def get_predictor_model(self):
        if self.model == None:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            self.model = model
        return self.model

    def predict_one(self, text):
        return self.model.predict([text]).tolist()

    def predict_many(self, texts):
        return self.model.predict(texts).tolist()
        