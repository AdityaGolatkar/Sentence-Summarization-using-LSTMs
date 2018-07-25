from __future__ import division
import pickle
import numpy as np
import pdb
import gensim
from gensim.models import Word2Vec


##############
## Training ##
##############

with open("../../Data/train_sentence.txt","rb") as fp:
    train_sentence = pickle.load(fp)

with open("../../Data/test_sentence.txt","rb") as fp:
    test_sentence = pickle.load(fp)


sentences = []

for i in range(len(train_sentence)):
    sentences_split = train_sentence[i].split()
    sentences.append(sentences_split)

for i in range(len(test_sentence)):
    sentence_split = test_sentence[i].split()
    sentences.append(sentence_split)

print("Total length of the corpus we have is :",len(sentences))

#model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True)
#model = gensim.models.Word2Vec.load_word2vec_format('../../GoogleNews-vectors-negative300.bin')
#print("Model loading done")

model = Word2Vec(sentences, min_count=1,size=300)
model.intersect_word2vec_format('../../GoogleNews-vectors-negative300.bin',lockf=1.0,binary=True)
model.train(sentences,total_examples=len(sentences),epochs=20)
model.wv.save_word2vec_format('../../google_finetuned.bin',binary=True)

#model = Word2Vec(sentences, min_count=1,size=256)
#model.wv.save_word2vec_format('../../my_word2vec.bin',binary=True)

pdb.set_trace()
