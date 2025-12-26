import pandas as pd
from src.utils import short_model_name

def test_short_model_name():
    assert short_model_name("Logistic Regression (Yahoo-trained)") == "LogReg"
    assert short_model_name("Random Forest (SEC-trained)") == "RF"
    assert short_model_name("Gradient Boosting (SEC-trained)") == "GB"
    assert short_model_name("Neural Network (Yahoo-trained)") == "NN"
