#this is just for making something simple idk.


##use correct libraries
import numpy as np
import pandas as pd
import tensorflow as tf
#print(tf.__version__)  # Check if TensorFlow is recognized

print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("COOK!!")


##Step 1 : Create a dataset and format for training
data = {
    'feature1': [0.1, 0.2,0.3,0.4,0.5],
    'feature2': [0.5,0.4,0.3,0.2,0.1],
    'label': [0,0,1,1,1]
}

df = pd.DataFrame(data)
X = df[['feature1', 'feature2']].values
y = df['label'].values


##Step 2: Create a Neural Network
    #
model = Sequential()
model.add(Dense(8, input_dim=2, activation= 'relu')) #hidden layer
model.add(Dense(1, activation='sigmoid')) #output layer

##Step 3: Compile the model
    #Specify loss loss fucntion, optimizer and metrics

model.compile(loss='binary_crossentropy', optimizer= 'adam'
              , metrics=['accuracy'])


##Step 5: Train the model
    #specify number -
    # Epoch (Full pass through entire dataset)
    # Batch size (number of samples processed in 1 go)
    # Verobose (0=nothing, 1=progress bar, 2=more information)

model.fit(X, y, epochs=100, batch_size=1, verbose=3)


##Step 6: Make predictions
    #process output

test_data = np.array([[0.2, 0.4]])
prediction = model.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)
