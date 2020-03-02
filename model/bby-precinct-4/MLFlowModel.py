from mlflow import pyfunc
import os
import pandas as pd

#Model wrapper that serves the predictions.
class MLFlowModel(object):

    def __init__(self):
        
        
        self.pyfunc_model = pyfunc.load_model("s3://ml-demo-dinesh/bestbuy/dd15d48ac01f454e91d22c7340ed56ce/artifacts/model")

    def predict(self,X,features_names):
        if not features_names is None and len(features_names)>0:
            df = pd.DataFrame(data=X,columns=features_names)
        else:
            df = pd.DataFrame(data=X)
        return self.pyfunc_model.predict(df)
