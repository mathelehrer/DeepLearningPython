import numpy as np

import network
from mnist import get_images
import matplotlib as plt

'''
lade Daten
'''

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

'''
Zeige Daten
'''

training_set, validation_set, test_set = mnist_loader.load_data()

def image2String(image):
    out = ''
    count=1
    for pix in image:
        if pix < 0.1:
            out += '_ '
        elif pix < 0.25:
            out += '- '
        elif pix < 0.5:
            out += '* '
        elif pix < 0.75:
            out += '% '
        else:
            out += 'X '
        if count%28==0:
            out += '\n'
        count+=1
    return out

def auswertung(ergebnis):
    maximum = -np.inf
    max_i=-1
    for i in range(10):
        if ergebnis[i][0]>maximum:
            maximum = ergebnis[i][0]
            max_i=i
    return max_i

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
test_data=list(test_data)

net = network.Network([784, 30, 10])
# ergebnis=net.feedforward(test_data[0][0])
# print(ergebnis)
# print(auswertung(ergebnis))

net.SGD(training_data, 1, 10, 3.0, test_data=test_data)

for i in range(20,30):
    print(test_set[1][i],":")
    print(image2String(test_set[0][i]))
    ergebnis = net.feedforward(test_data[i][0])
    print("Netzwerk vermutet: ",auswertung(ergebnis),"\n\n")





