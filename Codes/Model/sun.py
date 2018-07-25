from __future__ import print_function
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, LSTM, Dense, TimeDistributed, Bidirectional
import numpy as np
from keras.callbacks import ModelCheckpoint
import numpy as np
import pdb
import datetime
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "1"
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))


#no_train_samples = 15000
#no_test_samples = 500

no_train_samples = 20000
no_test_sample = 1000

epochs = 25
embedding_dim = 300
sentence_len = 25
batch_size = 8

###################
## Weights paths ##
###################

#expName = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

run = 5

best_weights = '../Weights/best_weights_exp_'+str(run)+'.h5'
last_weights = '../Weights/last_weights_exp_'+str(run)+'.h5'
initial_weights = '../Weights/best_weights_2.h5'

###############
## Load data ##
###############

train_encoder_input = np.load('../Data/refined_train_encoder_input.npy')
train_decoder_input = np.load('../Data/refined_train_decoder_input.npy')
train_decoder_output = np.load('../Data/refined_train_decoder_output.npy')
#pdb.set_trace()

######################
## Compressive LSTM ##
######################

drop = 0.2

# Define an input sequence and process it.
encoder_inputs = Input(shape=(sentence_len,embedding_dim))
encoder_1 = Bidirectional(LSTM(32, return_sequences=True,dropout=drop,recurrent_dropout=drop))(encoder_inputs)
encoder_2 = LSTM(64, return_sequences=True,dropout=drop,recurrent_dropout=drop)(encoder_1)
encoder_3 = LSTM(64, return_state=True,dropout=drop,recurrent_dropout=drop)(encoder_2)
encoder_outputs, state_h, state_c = encoder_3
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(sentence_len,embedding_dim))
decoder_1 = LSTM(64, return_sequences=True,return_state=True,dropout=drop,recurrent_dropout=drop)
decoder_1_output, _, _ = decoder_1(decoder_inputs,initial_state=encoder_states)
decoder_2 = LSTM(64, return_sequences=True,dropout=drop,recurrent_dropout=drop)(decoder_1_output)
decoder_3 = LSTM(64, return_sequences=True,dropout=drop,recurrent_dropout=drop)(decoder_2)
decoder_outputs = TimeDistributed(Dense(1,activation='sigmoid'))(decoder_3)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#sgd = SGD(lr=0.001, momentum=0.01, decay=0.0, nesterov=False)
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath= best_weights, monitor='val_acc',verbose=1, save_best_only=True,save_weights_only=True)

#class_weight = {0:0.3,1:0.7} 

def train():

    # Run training
    model.summary()
    model.fit([train_encoder_input, train_decoder_input], train_decoder_output,
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

    test_encoder_input = np.load('../Data/refined_test_encoder_input.npy')
    test_decoder_input = np.load('../Data/refined_test_decoder_input.npy')
    test_decoder_output = np.load('../Data/refined_test_decoder_output.npy')

    model.load_weights(last_weights)

    acc = model.evaluate([test_encoder_input,test_decoder_input],test_decoder_output,batch_size=None,verbose=1,steps=test_encoder_input.shape[0])
    print("Last weights acc: ",acc)

    model.load_weights(best_weights)

    acc = model.evaluate([test_encoder_input,test_decoder_input],test_decoder_output,batch_size=None,verbose=1,steps=test_encoder_input.shape[0])
    print("Best weights acc: ",acc)
    
    predictions = model.predict([test_encoder_input,test_decoder_input],batch_size=1,verbose=1)
    print(predictions.shape)
 
    np.save('../Data/Predictions.npy',predictions)


train()
test()




