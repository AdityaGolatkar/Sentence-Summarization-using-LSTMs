######################################################
## Remove the examples which have problems in them. ##
######################################################
from __future__ import division
import pickle
import numpy as np
import pdb

##############
## Training ##
##############

with open("../../Data/train_sentence.txt","rb") as fp:
    sentence = pickle.load(fp)

with open("../../Data/train_compressed_sentence.txt","rb") as fp:
    compressed_sentence = pickle.load(fp)

with open("../../Data/train_binary_output.txt","rb") as fp:
    bin_output = pickle.load(fp)

correct_ex = np.load('../../Data/train_correct_examples.npy')

###############################
## Filter too long sentences ##
###############################
sen_len = []
for i in range(len(bin_output)):
    sen_len.append(bin_output[i].shape[0])
sen_len = np.array(sen_len)

print("Mean Length of Sentences: ",np.mean(sen_len))
print("Std Length of Sentences: ",np.std(sen_len))

## We will not take examples with length greater than max_sen_len ##
#min_sen_len = int(np.mean(sen_len) - 1*np.std(sen_len))
#max_sen_len = int(np.mean(sen_len) + 1*np.std(sen_len))

min_sen_len = 20
max_sen_len = 25

sen_len_check = []
for i in range(len(bin_output)):
    if bin_output[i].shape[0] > max_sen_len or bin_output[i].shape[0] < min_sen_len:
        sen_len_check.append(0)
    else:
        sen_len_check.append(1)

refined_sen = []
refined_comp_sen = []
refined_bin_output = []
refined_len = []

for i in range(len(correct_ex)):
    if correct_ex[i] == 1 and sen_len_check[i] == 1:
        refined_sen.append(sentence[i])
        refined_comp_sen.append(compressed_sentence[i])
        refined_bin_output.append(bin_output[i])
        refined_len.append(bin_output[i].shape[0])

print("Max Length :", np.max(np.array(refined_len)))
#pdb.set_trace()

print("Total no of sentences within this range :", len(refined_sen))

with open("../../Data/refined_train_sentence.txt","wb") as fp:
    pickle.dump(refined_sen,fp)

with open("../../Data/refined_train_compressed_sentence.txt","wb") as fp:
    pickle.dump(refined_comp_sen,fp)

with open("../../Data/refined_train_binary_output.txt","wb") as fp:
    pickle.dump(refined_bin_output,fp)



#############
## Testing ##
#############


with open("../../Data/test_sentence.txt","rb") as fp:
    sentence = pickle.load(fp)

with open("../../Data/test_compressed_sentence.txt","rb") as fp:
    compressed_sentence = pickle.load(fp)

with open("../../Data/test_binary_output.txt","rb") as fp:
    bin_output = pickle.load(fp)

correct_ex = np.load('../../Data/test_correct_examples.npy')

###############################
## Filter too long sentences ##
###############################
sen_len = []
for i in range(len(bin_output)):
    sen_len.append(bin_output[i].shape[0])
sen_len = np.array(sen_len)

print("Mean Length of Sentences: ",np.mean(sen_len))
print("Std Length of Sentences: ",np.std(sen_len))

## We will not take examples with length greater than max_sen_len ##
#min_sen_len = int(np.mean(sen_len) - 1*np.std(sen_len))
#max_sen_len = int(np.mean(sen_len) + 1*np.std(sen_len))

min_sen_len = 20
max_sen_len = 25

sen_len_check = []
for i in range(len(bin_output)):
    if bin_output[i].shape[0] > max_sen_len:
        sen_len_check.append(0)
    else:
        sen_len_check.append(1)
        


refined_sen = []
refined_comp_sen = []
refined_bin_output = []
refined_len = []

for i in range(len(correct_ex)):
    if correct_ex[i] == 1 and sen_len_check[i] == 1:
        refined_sen.append(sentence[i])
        refined_comp_sen.append(compressed_sentence[i])
        refined_bin_output.append(bin_output[i])
        refined_len.append(bin_output[i].shape[0])
#pdb.set_trace()

print("Total no of sentences in this range:", len(refined_sen))

print("Max Length: ",np.max(np.array(refined_len)))
with open("../../Data/refined_test_sentence.txt","wb") as fp:
    pickle.dump(refined_sen,fp)

with open("../../Data/refined_test_compressed_sentence.txt","wb") as fp:
    pickle.dump(refined_comp_sen,fp)

with open("../../Data/refined_test_binary_output.txt","wb") as fp:
    pickle.dump(refined_bin_output,fp)



