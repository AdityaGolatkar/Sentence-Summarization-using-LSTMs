from __future__ import print_function
#from keras.models import Model
#from keras.optimizers import SGD
#from keras import regularizers
#from keras.layers import Input, LSTM, Dense, TimeDistributed, Bidirectional
import numpy as np
#from keras.callbacks import ModelCheckpoint
import numpy as np
import pdb
import datetime
#import tensorflow as tf

"""
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "1"
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
"""

no_train_samples = 20000
no_test_samples = 1000
epochs = 25
embedding_dim = 300
sentence_len = 25
batch_size = 8

###################
## Weights paths ##
###################

#expName = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

run = 6

best_weights = '../../Weights/pure_best_weights_exp_'+str(run)+'.h5'
last_weights = '../../Weights/pure_last_weights_exp_'+str(run)+'.h5'
initial_weights = '../../Weights/best_weights_2.h5'

###############
## Load data ##
###############

train_encoder_input = np.load('../../Data/refined_train_encoder_input.npy')
train_decoder_input = np.load('../../Data/refined_train_decoder_input.npy')
train_decoder_output = np.load('../../Data/refined_train_decoder_output.npy')
train_arora_input = np.load('../../Data/refined_train_arora_hidden_state.npy')
train_arora_input = train_arora_input.T

#train_input = []
train_input_comp = []

for i in range(no_train_samples):
    train_input = []
    for j in range(sentence_len):
        train_input.append(np.append(train_decoder_input[i,j,:].reshape((1,300)),train_arora_input[i,:]))
        
    train_input_comp.append(np.array(train_input))
#pdb.set_trace()
train_input_comp = np.array(train_input_comp)
print(train_input_comp.shape)

pdb.set_trace()

######################
## Compressive LSTM ##i
######################

drop = 0.2

inputs = Input(shape=(sentence_len,embedding_dim))
lstm_1 = LSTM(64, return_sequences=True,dropout=drop,recurrent_dropout=drop,
kernel_regularizer=regularizers.l2(0.01))(inputs)

lstm_2 = Bidirectional(LSTM(32, return_sequences=True,dropout=drop,
recurrent_dropout=drop,kernel_regularizer=regularizers.l2(0.01)))(lstm_1_output)

lstm_3 = Bidirectional(LSTM(32, return_sequences=True,dropout=drop,
recurrent_dropout=drop,kernel_regularizer=regularizers.l2(0.01)))(lstm_2)

dense_1 = TimeDistributed(Dense(1,activation='sigmoid'))(lstm_3)

model = Model(inputs,dense_1)
#sgd = SGD(lr=0.001, momentum=0.01, decay=0.0, nesterov=False)
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath= best_weights, monitor='val_acc',verbose=1, save_best_only=True,save_weights_only=True)

#class_weight = {0:0.3,1:0.7} 


def train():

    # Run training
    model.summary()
    model.fit(train_decoder_input, train_decoder_output,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          #class_weight=class_weight,
          callbacks=[checkpointer])

    #del train_encoder_input,train_decoder_input,train_decoder_output
    model.save_weights(last_weights)


def test():

####################
## Load test data ##
####################

    test_decoder_input = np.load('../../Data/refined_test_decoder_input.npy')
    test_decoder_output = np.load('../../Data/refined_test_decoder_output.npy')
    test_arora_input = np.load('../../Data/refined_test_arora_hidden_state.npy')
    test_arora_input = np.reshape(test_arora_input,(no_test_samples,1,embedding_dim))

    model.load_weights(last_weights)

    acc = model.evaluate([test_decoder_input,test_arora_input],test_decoder_output,batch_size=None,verbose=1,steps=test_decoder_output.shape[0])
    print("Last weights acc: ",acc)

    model.load_weights(best_weights)

    acc = model.evaluate([test_decoder_input,test_arora_input],test_decoder_output,batch_size=None,verbose=1,steps=test_decoder_output.shape[0])
    print("Best weights acc: ",acc)

    predictions = model.predict([test_decoder_input,test_arora_input],batch_size=1,verbose=1)
    print(predictions.shape)

    np.save('../../Data/pure_Predictions.npy',predictions)


train()
test()



