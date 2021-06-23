#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd


# In[2]:


ds = pd.read_csv("train.csv")
df = ds.sample(frac=1)


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


print((df.valor==1).sum())#eletrica
print((df.valor==0).sum())#direito


# In[6]:


import re
import string

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"",text)

def remove_punct(text):
    translator = str.maketrans("","",'!""#$%&\'()*+,./:;<=>?@[\\]^_`{|}~º')
    return text.translate(translator)

def remove_hifen(text):
    translator = str.maketrans('-',' ')
    return text.translate(translator)

string.punctuation


# In[7]:


pattern = re.compile(r"https?//(\S+|www)\.\S+")
for t in df.texto:
    matches = pattern.findall(t)
    for match in  matches:
        print(t)
        print(match)
        print(pattern.sub(r"",t))
        
    if len(matches)> 0:
        break


# In[8]:


df["texto"] = df.texto.map(remove_URL)
df["texto"] = df.texto.map(remove_punct)
df["texto"] = df.texto.map(remove_hifen)


# In[9]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop = set(stopwords.words("portuguese"))

def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


# In[10]:


stop


# In[11]:


df["texto"] = df.texto.map(remove_stopwords)


# In[12]:


df.texto


# In[13]:


df.texto[72]


# In[14]:


from collections import Counter

def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count

counter = counter_word(df.texto)


# In[15]:


len(counter)


# In[16]:


counter.most_common(5)


# In[17]:


num_unique_words = len(counter)


# In[18]:


train_size = 56

train_df = df[:train_size]
val_df = df[train_size:]


# In[19]:


print(len(train_df))
print(len(val_df))


# In[20]:


train_sentences = train_df.texto.to_numpy()
train_labels = train_df.valor.to_numpy()

val_sentences = val_df.texto.to_numpy()
val_labels = val_df.valor.to_numpy()


# In[21]:


train_sentences.shape, val_sentences.shape


# In[22]:


from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = num_unique_words,oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)


# In[23]:


word_index = tokenizer.word_index


# In[24]:


word_index


# In[25]:


train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)


# In[26]:


#print(train_sentences[10:15])
#print(train_sequences[10:15])


# In[27]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 100

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding = "post",truncating = "post")
val_padded = pad_sequences(val_sequences, maxlen = max_length, padding = "post", truncating = "post")
train_padded.shape, val_padded.shape


# In[28]:


train_padded[10]


# In[29]:


#print(train_sentences[10])
#print(train_sequences[10])
#print(train_padded[10])


# In[30]:


reverse_word_index = dict([(idx,word) for (word, idx) in word_index.items()])


# In[31]:


reverse_word_index


# In[32]:


def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])


# In[33]:


decoded_text = decode(train_sequences[10])

#print(train_sequences[10])
#print(decoded_text)


# In[63]:


from tensorflow.keras import layers

model = keras.models.Sequential()
model.add(layers.Embedding(num_unique_words, 32, input_length = max_length))

model.add(layers.LSTM(256,dropout = 0.1))
#model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(24, activation = "relu"))
model.add(layers.Dense(24, activation = "relu"))
#model.add(layers.Dense(24, activation = "softmax"))
#model.add(layers.Softmax())
model.add(layers.Dense(1, activation = "sigmoid"))


model.summary()


# In[64]:


loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer = optim, metrics = metrics)


# In[65]:


import time
start = time.perf_counter()
model.fit(train_padded,train_labels, epochs = 20, validation_data=(val_padded,val_labels), verbose=2)
finish = time.perf_counter()
print(f'\nFinished in {round(finish-start, 2)} second(s)')


# In[66]:


predictions = model.predict(val_padded)
print(predictions)
predictions = [1 if p > 0.5 else 0 for p in predictions]


# In[67]:


index = 7
print(val_sentences[index])
print(val_padded[index])
print("Label: ",val_labels[index])
print("Resultado: ",predictions[index])
print(val_labels)
print(predictions)


# In[68]:


df_t = pd.read_csv("eval.csv")


# In[69]:


df_t.head()


# In[70]:


def make_test():
    df_t["texto"] = df_t.texto.map(remove_URL)
    df_t["texto"] = df_t.texto.map(remove_punct)
    df_t["texto"] = df_t.texto.map(remove_hifen)
    df_t["texto"] = df_t.texto.map(remove_stopwords)


# In[71]:


def pat():
    for t in df_t.texto:
        matches = pattern.findall(t)
        for match in  matches:
            print(t)
            print(match)
            print(pattern.sub(r"",t))        
        if len(matches)> 0:
            break


# In[72]:


pat()
make_test()


# In[73]:


df_t.texto


# In[74]:


test_sentences = df_t.texto.to_numpy()
test_labels = df_t.valor.to_numpy()


# In[75]:


test_sequences = tokenizer.texts_to_sequences(test_sentences)
word_index_test = tokenizer.word_index
word_index_test


# In[76]:


print(test_sentences)
print(test_sequences)


# In[77]:


test_padded = pad_sequences(test_sequences, maxlen=max_length, padding = "post",truncating = "post")


# In[78]:


print(test_padded)


# In[79]:


predictions_t = model.predict(test_padded)
predictions_t = [1 if p > 0.5 else 0 for p in predictions_t]


# In[80]:


print(predictions_t)
print(test_labels)


# In[81]:


decoded_test = decode(test_sequences[7])


# In[82]:


print(decoded_test)


# In[83]:


#model.save_weights('m_salvo/')


# In[84]:


#model.save('modelo_completo/')


# In[ ]:





# In[ ]:




