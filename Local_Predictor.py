from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.util import import_submodules

class LocalPredictor():
    def __init__(self, model_dir="../scicite-pretrained.tar.gz", predictor_type="predictor_scicite",include_package="scicite", overrides=""):
        import_submodules(include_package)
        self.model_archive = load_archive(model_dir, overrides=overrides)

        self.predictor = Predictor.from_archive(self.model_archive, predictor_type)


    def predict(self, text, sectionName="None", label="background", citingPaperId='0', citedPaperId='0', excerpt_index=1):
        input_text = self.input_format(text, sectionName, label, citingPaperId, citedPaperId, excerpt_index)
        prediction = self.predictor.predict_json(input_text)

        predicted_label = prediction["prediction"]

        #print(prediction)

        return predicted_label
    
    def input_format(self, text, sectionName, label, citingPaperId, citedPaperId, excerpt_index):
        return {"sectionName": sectionName, "string": text, "label": label, "citingPaperId": citingPaperId, "citedPaperId": citedPaperId, "excerpt_index": excerpt_index}



def main():
    input_text = "However, how frataxin interacts with the Fe-S cluster biosynthesis components remains unclear as direct one-to-one interactions with each component were reported (IscS [12,22], IscU/Isu1 [6,11,16] or ISD11/Isd11 [14,15])."

    
    a = LocalPredictor("./data/scicite-pretrained.tar.gz")
    print(f"Prediction: {a.predict(input_text)}")

if __name__ == "__main__":
    main()


'''
{'citingPaperId': '0', 'citedPaperId': '0', 'citation_id': None, 'probabilities': array([9.9999130e-01, 8.1414037e-06, 6.4103085e-07], dtype=float32), 
'prediction': 'background', 'original_label': 'background', 
'citation_text': ['However', ',', 'how', 'frataxin', 'interacts', 'with', 'the', 'Fe', '-', 'S', 'cluster', 'biosynthesis', 'components', 'remains', 'unclear', 'as',
 'direct', 'one', '-', 'to', '-', 'one', 'interactions', 'with', 'each', 'component', 'were', 'reported', '(', 'IscS', '[', '12,22', ']', ',', 'IscU', '/', 'Isu1',
   '[', '6,11,16', ']', 'or', 'ISD11/Isd11', '[', '14,15', ']', ')', '.'],
    
      'attention_dist': array([0.02477127, 0.03463863, 0.0258912 , 0.00947763, 0.01730575,
       0.01359555, 0.01667009, 0.0154144 , 0.01352241, 0.0096101 ,
       0.01402801, 0.01619877, 0.019249  , 0.02374622, 0.03780404,
       0.04399198, 0.03830929, 0.0316029 , 0.03163977, 0.03158367,
       0.02779947, 0.03238247, 0.027984  , 0.01536759, 0.01143561,
       0.01331056, 0.03449196, 0.03662481, 0.03409727, 0.04304172,
       0.02829501, 0.03360318, 0.02156637, 0.02288388, 0.01192443,
       0.01126496, 0.01214793, 0.01259646, 0.01438388, 0.0077909 ,
       0.01346644, 0.01002054, 0.01262495, 0.00884409, 0.00633101,
       0.01254504, 0.01412472], dtype=float32)}'''