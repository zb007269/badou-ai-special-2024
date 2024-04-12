import numpy
import scipy.special
class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))