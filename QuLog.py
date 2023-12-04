import pandas as pd
from keras.preprocessing.text import Tokenizer
#import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Embedding
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
sns.set()
def remove_stopwords(text):
    text = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word.isalpha() and not word in stop_words]
    return ''.join(text)


df = pd.read_csv('C:/Course_work/DSCI_644/Project/QuLog-main/Group_2/Data_In/nine_systems_data.csv')
df = df.drop(df[(df.log_level != "info") & (df.log_level != "warn") & (df.log_level != "error")].index)
x = df.apply(lambda row: remove_stopwords(row['static_text']), axis=1)
y= df['log_level'].replace('info', 0.0).replace('error', 1.0).replace('warn', 2.0)

max_words = 200
max_length = 4


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)
x= pad_sequences(sequences, maxlen=max_length)


model = Sequential()
model.add(Embedding(max_words, 5, input_length=max_length))
model.add(Flatten())
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(x, y, validation_split=0.2, epochs=5, batch_size=20)

preds = model.predict
'''
accuracy = hist.history['accuracy']
val = hist.history['val_accuracy']
epochs = range(1, len(accuracy)+1)

plt.plot(epochs, accuracy, '-', label='Training accuracy')
plt.title('Train and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()
'''