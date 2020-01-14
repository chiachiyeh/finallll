from . import svhn2mnist
from . import usps
from . import syn2gtrsb
#import syndig2svhn
from . import real_object_MCD_DA


def Generator(source, target, pixelda=False):
    if source == 'usps' or target == 'usps':
        return usps.Feature()
    elif source == 'svhn':
        return svhn2mnist.Feature()
    elif source == 'synth':
        return syn2gtrsb.Feature()
    elif source == 'object':
        return real_object_MCD_DA.Feature()


def Classifier(source, target):
    if source == 'usps' or target == 'usps':
        return usps.Predictor()
    if source == 'svhn':
        return svhn2mnist.Predictor()
    if source == 'synth':
        return syn2gtrsb.Predictor()
    if source == 'object':
        return real_object_MCD_DA.Predictor()
