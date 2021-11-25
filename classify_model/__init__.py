from classify_model.lenet import lenet
from classify_model.googlenet import googlenet
from classify_model.mobilenet import mobilenetv2

def make_model(model_name):
    if model_name =='lenet':
        return lenet()
    if model_name == 'googlenet':
        return googlenet()
    if model_name == 'mobilenet':
        return mobilenetv2()