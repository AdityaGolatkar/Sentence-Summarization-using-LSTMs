from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed
import numpy as np
from keras.callbacks import ModelCheckpoint
import pdb

no_train_samples = 20000
no_test_samples = 2000
epochs = 50
embedding_dim = 300
sentence_len = 60
batch_size = 32

###################
## Weights paths ##
###################

best_weights = '../Weights/best_weights_1.h5'
last_weights = '../Weights/last_weights_1.h5'
initial_weights = '../Weights/best_weights_1.h5'

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


# Define an input sequence and process it.
encoder_inputs = Input(shape=(sentence_len,embedding_dim))
encoder_1 = LSTM(300, return_sequences=True)(encoder_inputs)
encoder_2 = LSTM(30, return_sequences=True)(encoder_1)
encoder_3 = LSTM(30, return_state=True)(encoder_2)
encoder_outputs, state_h, state_c = encoder_3
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(sentence_len,embedding_dim))
decoder_1 = LSTM(30, return_sequences=True)
decoder_1_1 = decoder_1(decoder_inputs,initial_state=encoder_states)
decoder_2 = LSTM(30, return_sequences=True)(decoder_1_1)
decoder_3 = LSTM(30, return_sequences=True)(decoder_2)
decoder_outputs = TimeDistributed(Dense(1,activation='sigmoid'))(decoder_3)
#Decoder = Model(inputs=decoder_inputs,outputs=decoder_4)
#decoder_outputs = Decoder(decoder_inputs,initial_state=encoder_states)


"""
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,embedding_dim))
encoder = LSTM(30, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,embedding_dim))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(30, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(1, activation='sigmoid')
decoder_outputs = decoder_dense(decoder_outputs)
"""

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

checkpointer = ModelCheckpoint(filepath= best_weights, monitor='val_acc',verbose=0, save_best_only=True,save_weights_only=True)

# Run training
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit([train_encoder_input, train_decoder_input], train_decoder_output,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


model.save_weights(last_weights)

del encoder_input,decoder_input,decoder_output

####################
## Load test data ##
####################

test_encoder_input = np.load('../Data/refined_test_encoder_input.npy')
test_decoder_input = np.load('./Data/refined_test_decoder_input.npy')
test_decoder_output = np.load('../Data/refined_test_decoder_output.npy')

acc = model.evaluate([test_encoder_input,test_decoder_input],test_decoder_output,batch_size=1,shuffle=False)
print("Last weights acc: ",acc)

model.load_weights(best_weights)

acc = model.evaluate([test_encoder_input,test_decoder_input],test_decoder_output,batch_size=1,shuffle=False)
print("Best weights acc: ",acc)








