import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

border = "-"*50
#---------------------------------------------------
#Step 1 : Load the data
#---------------------------------------------------
train_Sentences = [
    "food was good",
    "food was bad",
    "food was excellent",
    "food was terrible",
    "service was good",
    "service was bad",
    "service was excellent",
    "service was terrible",
    "ambience was good",
    "ambience was bad",
    "ambience was excellent",
    "ambience was terrible"
]

train_labels = [
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0
]
#---------------------------------------------------
#Step 2 : Tokenization
#---------------------------------------------------
tokenizer = Tokenizer(oov_token = "<OOV>")
tokenizer.fit_on_texts(train_Sentences)
#---------------------------------------------------
#Step 3 : Convert Training data into sequence
#---------------------------------------------------
train_sequence = tokenizer.texts_to_sequences(train_Sentences)

print(border)
print("Training Sequences : ")
print(border)

for sentance, sequence in zip(train_Sentences,train_sequence):
    print(sentance,"->",sequence)
print(border)
#---------------------------------------------------
#step 4 : Apply Padding
#---------------------------------------------------
max_length = 4

X_train = pad_sequences(
    train_sequence,
    maxlen = max_length,
    padding = 'pre'
)

Y_train = np.array(train_labels)

print(border)
print("Padded training data ")
print(border)

print(X_train)
print(border)

print(border)
print("Training Labels : ")
print(border)

print(Y_train)
print(border)
#---------------------------------------------------
#Step 5 : Calculate Vocabulary size
#---------------------------------------------------
vocab_size = len(tokenizer.word_index) + 1

print(border)
print("Vocabu;ary size is : ",vocab_size)
print(border)
#---------------------------------------------------
#Step 6 : Build RNN Model
#---------------------------------------------------
print(border)
print("Model Building")
print(border)
model = Sequential()

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=8,
        input_length=max_length
    )
)

model.add(
    SimpleRNN(
        units = 8,
        activation = "tanh"
    )
)

model.add(
    Dense(
        units=1,
        activation="sigmoid"
    )
)
#---------------------------------------------------
#Step 7 : Complete the model
#---------------------------------------------------
model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)
#---------------------------------------------------
#Step 8 : Display model
#---------------------------------------------------
model.build(input_shape = (None,max_length))

print(border)
print("Model Architecture")
print(border)

model.summary()
print(border)
#---------------------------------------------------
#step 9 : Train the model
#---------------------------------------------------
history = model.fit(
    X_train,
    Y_train,
    epochs = 100,
    verbose =1
)
print(border)
print("MOdel training completeed")
print(border)
#---------------------------------------------------
#Step 10 : create unseen data
#---------------------------------------------------
test_sentences = [
    "Service was amezing",
    "Service was horrible",
    "experiecnce was excellent",
    "experience was terrible"
]
#---------------------------------------------------
#step 11 : convert text to sequence
#---------------------------------------------------
test_sequences = tokenizer.texts_to_sequences(test_sentences)

X_test = pad_sequences(
    test_sequences,
    maxlen = max_length,
    padding = "pre"
)
#---------------------------------------------------
#Step 12 : Predict 
#---------------------------------------------------
for text, sequence,padded in zip(test_sentences,test_sequences,X_test):
    input_data = np.array([padded])

    prediction = model.predict(input_data,verbose=0)

    probability =float(prediction[0][0])

    print("Sentence : ",text)
    print("Sequence : ",sequence)
    print("Prediction : ",probability)

    if(probability >= 0.5):
        print('Sentiment : Positvie')
    else:
        print("Sentiment : Negative")