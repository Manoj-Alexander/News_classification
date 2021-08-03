import pickle

class news:
    def __init__(self,text):

        self.text=text
    def predictionnews(self):
        model=pickle.load(open("final_model.pkl","rb"))

        return model.predict(self.text)

