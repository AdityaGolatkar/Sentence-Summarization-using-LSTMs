#####################################################
## Takes the test and the compressed sentence to   ##
## generate a binary vector output for a sentence. ##
#####################################################
from __future__ import division
import numpy as np
import pickle

##############
## Training ##
##############

with open("../Data/train_sentence.txt","rb") as fp:
    sentence = pickle.load(fp)

with open("../Data/train_compressed_sentence.txt","rb") as fp:
    compressed_sentence = pickle.load(fp)

binary_output = []
correct_ex = []

for i in range(len(sentence)):
    bin_vec = np.zeros((len(sentence[i].split()),1))
    words_in_sentence = sentence[i].split()
    words_in_comp_sent = compressed_sentence[i].split()
    #print(sentence[i])
    #print(compressed_sentence[i])
    #print(words_in_sentence)
    #print(words_in_comp_sent)
    correct_ex.append(1)
    for j in range(len(words_in_comp_sent)):
        try:
            bin_vec[words_in_sentence.index(words_in_comp_sent[j])] = 1

        except:
            try:
                bin_vec[words_in_sentence.index(words_in_comp_sent[j].lower())] = 1
            except:
                #print("problem")
                correct_ex[-1] = 0
                break

    binary_output.append(bin_vec)
    print(sum(correct_ex),i)
    #print(i)    
    #print(sentence[i])
    #print(bin_vec)
    #print(compressed_sentence[i])    

with open("../Data/train_binary_output.txt","wb") as fp: #Binary output for the sentences.
    pickle.dump(binary_output,fp)

np.save('../Data/train_correct_examples.npy',correct_ex) #Index of Sentences which are correct.


#############
## Testing ##
#############

with open("../Data/test_sentence.txt","rb") as fp:
    sentence = pickle.load(fp)

with open("../Data/test_compressed_sentence.txt","rb") as fp:
    compressed_sentence = pickle.load(fp)

binary_output = []
correct_ex = []

for i in range(len(sentence)):
    bin_vec = np.zeros((len(sentence[i].split()),1))
    words_in_sentence = sentence[i].split()
    words_in_comp_sent = compressed_sentence[i].split()
    #print(sentence[i])
    #print(compressed_sentence[i])
    #print(words_in_sentence)
    #print(words_in_comp_sent)
    correct_ex.append(1)
    for j in range(len(words_in_comp_sent)):
        try:
            bin_vec[words_in_sentence.index(words_in_comp_sent[j])] = 1
            
        except:
            try:
                bin_vec[words_in_sentence.index(words_in_comp_sent[j].lower())] = 1
            except:
                #print("problem")
                correct_ex[-1] = 0
                break

    binary_output.append(bin_vec)
    print(sum(correct_ex),i)    
    #print(i)    
    #print(sentence[i])
    #print(bin_vec)
    #print(compressed_sentence[i])    

with open("../Data/test_binary_output.txt","wb") as fp: #Binary output for the sentences.
    pickle.dump(binary_output,fp)

np.save('../Data/test_correct_examples.npy',correct_ex) #Index of Sentences which are correct.
        
