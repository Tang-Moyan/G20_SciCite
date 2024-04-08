from allennlp.predictors import Predictor

class Local_Predictor():
    def __init__(self, model_dir="../model/scicite/"):
        self.predictor = Predictor.from_path(model_dir)

    def predict_input(self, text):
        prediction = self.predictor.predict(string=text)

        predicted_label = prediction["label"]

        return predicted_label
