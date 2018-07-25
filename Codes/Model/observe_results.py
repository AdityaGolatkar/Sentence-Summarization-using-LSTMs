from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
from sklearn.metrics import f1_score
import pdb

test_decoder_output = np.load('../Data/refined_test_decoder_output.npy')

with open('../Data/refined_test_sentence.txt','rb') as f:
    test_sentence = pickle.load(f)

test_sentence = test_sentence[0:1000]

with open('../Data/refined_test_compressed_sentence.txt','rb') as f:
    test_compressed_sentence = pickle.load(f)

test_compressed_sentence = test_compressed_sentence[0:1000]

threshold = 0.5

predictions = np.load('../Data/pure_Predictions.npy')
#predictions = np.load('../Data/Predictions.npy')
predictions[predictions>threshold] = 1
predictions[predictions<=threshold] = 0


if predictions.shape[0] != len(test_sentence):
    print("Kuch toh pain hai")
    pdb.set_trace()


average_f1 = 0
prediction_sentence = []

for i in range(predictions.shape[0]):
    average_f1+=f1_score(predictions[i,:],test_decoder_output[i,:])    
    sentence = test_sentence[i].split()
    #print(sentence)
    pred_sen = []
    for j in range(len(sentence)):
        if predictions[i,j] == 1:
            pred_sen.append(sentence[j])
    prediction_sentence.append(pred_sen)
    
    print(i)

print("Average F1 score = ",average_f1/predictions.shape[0])



log = open("destiny.txt","w")

for i in range(predictions.shape[0]):
    
    print("Actual Sentence :",file=log)
    print(test_sentence[i],file=log)
    print("Actual Summary :",file=log)
    print(test_compressed_sentence[i],file=log)
    print("Our summary :",file=log)
    print(prediction_sentence[i],file=log)
    print("",file=log)









