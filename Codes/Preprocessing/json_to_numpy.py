from __future__ import division
import numpy as np
import pdb
import pickle
import re

##########################
## Extracting sentences ##
##########################

sentence = []
compressed_sentence = []

ct = -1

##############
## Training ##
##############

for file_no in range(10):
    
    if file_no < 9:
        filename = '/home/audy/Desktop/AML_Project/sentence-compression/data/sent-comp.train0'+str(file_no+1)+'.json' 
    else:
       filename = '/home/audy/Desktop/AML_Project/sentence-compression/data/sent-comp.train'+str(file_no+1)+'.json'
    
    #filename = '/home/audy/Desktop/AML_Project/sentence-compression/data/comp-data.eval.json'
    with open(filename,'r') as f:
        file_data = f.readlines()
        #pdb.set_trace()
        for i in range(len(file_data)):
            if file_data[i][0:15] == '    "sentence":':
                ct=ct+1
                if len(sentence) > 0 and ct%2 != 0:
                    continue
                else:
                    sentence.append(file_data[i][17:-4])            
                
                sentence[-1] = re.sub('[^0-9a-zA-Z]+', ' ', sentence[-1])

                print(len(sentence),"sentence")
                #print(sentence)
                #pdb.set_trace()
                #print(sentence[len(sentence)-1])
            elif file_data[i][0:11] == '    "text":' and len(file_data[i-1]) == 19:#file_data[i-1][0:16] == '  "compression"':
                compressed_sentence.append(file_data[i][13:-4])
                
                compressed_sentence[-1] = re.sub('[^0-9a-zA-Z]+', ' ', compressed_sentence[-1])



                print(len(compressed_sentence),"compressed_sentence")
                #print(compressed_sentence[len(compressed_sentence)-1])

#pdb.set_trace()

											
with open('../Data/train_sentence.txt','wb') as fp:
    pickle.dump(sentence,fp)

with open('../Data/train_compressed_sentence.txt','wb') as fp:
    pickle.dump(compressed_sentence,fp) 			
	

#############
## Testing ##
#############

sentence = []
compressed_sentence = []

ct = -1


for file_no in range(1):
    
    filename = '/home/audy/Desktop/AML_Project/sentence-compression/data/comp-data.eval.json'
    with open(filename,'r') as f:
        file_data = f.readlines()
        #pdb.set_trace()
        for i in range(len(file_data)):
            if file_data[i][0:15] == '    "sentence":':
                ct=ct+1
                if len(sentence) > 0 and ct%2 != 0:
                    continue
                else:
                    sentence.append(file_data[i][17:-4])            
                
                sentence[-1] = re.sub('[^0-9a-zA-Z]+', ' ', sentence[-1])

                print(len(sentence),"sentence")
                #print(sentence)
                #pdb.set_trace()
                #print(sentence[len(sentence)-1])
            elif file_data[i][0:11] == '    "text":' and len(file_data[i-1]) == 19:#file_data[i-1][0:16] == '  "compression"':
                compressed_sentence.append(file_data[i][13:-4])
                
                compressed_sentence[-1] = re.sub('[^0-9a-zA-Z]+', ' ', compressed_sentence[-1])



                print(len(compressed_sentence),"compressed_sentence")
                #print(compressed_sentence[len(compressed_sentence)-1])

#pdb.set_trace()



with open('../Data/test_sentence.txt','wb') as fp:
    pickle.dump(sentence,fp)

with open('../Data/test_compressed_sentence.txt','wb') as fp:
    pickle.dump(compressed_sentence,fp)			
	
																																																																																																																																											
