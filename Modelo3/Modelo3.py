#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import re
import string
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
import time


# In[2]:


def get_data(nome_arquivo,shuffle):
    ds = pd.read_csv(nome_arquivo,encoding="utf-8")
    if shuffle:
        ds = ds.sample(frac=1)
    ds['texto'] = ds['texto'].apply(str)
    return ds


# In[3]:


stop = set(stopwords.words("portuguese"))

def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)

def remove_URL(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub(r"",text)

def remove_punct(text):
    translator = str.maketrans("","",'!""#$%&\'()*+,./:;<=>?@[\\]^_`{|}~º')
    translator = str.maketrans("","",'!""#$%&\'()*+,./:;<=>?@[\\]^_`{|}~º')
    return text.translate(translator)

def remove_numbers(text):
    result = ''.join([i for i in text if not i.isdigit()])
    return result

def remove_hifen(text):
    translator = str.maketrans('-',' ')
    return text.translate(translator)


# In[4]:


pattern = re.compile(r"https?//(\S+|www)\.\S+")
def pat(df_t):
    for t in df_t.texto:
        matches = pattern.findall(t)
        for match in  matches:
            print(t)
            print(match)
            print(pattern.sub(r"",t))        
        if len(matches)> 0:
            break


# In[5]:


def make_test(df_t):
    df_t["texto"] = df_t.texto.map(remove_URL)
    df_t["texto"] = df_t.texto.map(remove_punct)
    df_t["texto"] = df_t.texto.map(remove_hifen)
    #df_t["texto"] = df_t.texto.map(remove_numbers)
    df_t["texto"] = df_t.texto.map(remove_stopwords)


# In[6]:


def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


# In[7]:


def data_split(df,size):
    train_size = int(df.shape[0]*size)
    train_df = df[:train_size]
    val_df = df[train_size:]
    return train_df, val_df


# In[8]:


def data_to_numpy(df):    
    train_sentences = train_df.texto.to_numpy()
    train_labels = train_df.valor.to_numpy()
    val_sentences = val_df.texto.to_numpy()
    val_labels = val_df.valor.to_numpy()
    return train_sentences, train_labels, val_sentences, val_labels


# In[9]:


def prepare(teste):
    teste = remove_URL(teste)
    teste = remove_punct(teste)
    teste = remove_hifen(teste)
    teste = remove_stopwords(teste)    
    return teste

def predict(teste):
    predictions = model.predict(np.array(teste)) 
    p1 = [np.argmax(element) for element in predictions]
    if p1[0]:
        print("Disciplina: Eletronica")
    else:
        print("Disciplina: Elétrica")
    return predictions,p1

def tokenization(df):      
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    return train_sequences, val_sequences, word_index

def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])

def precision(test_labels,predictions):
    counter = len(test_labels)
    list_c = [i for i,j in zip(predictions,test_labels) if i == j]
    return len(list_c)/counter*100


# In[10]:


df = get_data('train.csv',1)


# In[11]:


df.shape


# In[12]:


df.head()


# In[13]:


print((df.valor==1).sum())#eletronica
print((df.valor==2).sum())#direito
print((df.valor==3).sum())#eletrica
print((df.valor==4).sum())#odontologia
print((df.valor==5).sum())#computação
print((df.valor==6).sum())#geografia
print((df.valor==7).sum())#ambiental
print((df.valor==8).sum())#mecanica


# In[14]:


pat(df)
make_test(df)


# In[15]:


counter = counter_word(df.texto)
num_unique_words = len(counter)
counter.most_common(5)


# In[16]:


train_df, val_df = data_split(df,0.8)

print(len(train_df))
print(len(val_df))


# In[17]:


train_sentences, train_labels, val_sentences, val_labels = data_to_numpy(df)
train_sentences.shape, val_sentences.shape


# In[18]:


tokenizer = Tokenizer(num_words = num_unique_words,oov_token="<OOV>")
train_sequences, val_sequences, word_index = tokenization(df)


# In[19]:


max_length = 500

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding = "post",truncating = "post")
val_padded = pad_sequences(val_sequences, maxlen = max_length, padding = "post", truncating = "post")
train_padded.shape, val_padded.shape


# In[20]:


reverse_word_index = dict([(idx,word) for (word, idx) in word_index.items()])


# In[21]:


#model.add(layers.LSTM(32,dropout = 0.1))
#model.add(layers.Conv1D(128,1,activation='relu'))
#model.add(layers.Dense(128, activation = "relu"))
#model.add(layers.Bidirectional(layers.LSTM(32)))
#model.add(layers.Dense(720, activation = "relu"))
#model.add(layers.Dense(128, activation = "relu"))


# In[36]:


model = keras.models.Sequential()

model.add(layers.Embedding(num_unique_words, 32, input_length = max_length))

model.add(layers.GlobalAveragePooling1D())

model.add(layers.Dense(72,activation = "relu"))

model.add(layers.Dense(9, activation = "softmax"))

model.summary()


# In[37]:


loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer = optim, metrics = metrics)


# In[38]:


start = time.perf_counter()
model.fit(train_padded,train_labels, epochs = 50, validation_data=(val_padded,val_labels), verbose=2)
finish = time.perf_counter()
print(f'\nFinished in {round(finish-start, 2)} second(s)')


# In[39]:


predictions = model.predict(val_padded)
predictions = [np.argmax(element) for element in predictions]


# In[40]:


#print("Sentença: ",val_sentences[0])
print("Label: ",val_labels[0])
print("Resultado: ",predictions[0],'\n')
print(val_labels,'\n')
print(predictions)


# In[41]:


r = precision(val_labels,predictions)
print(r)


# In[42]:


ds = get_data('eval.csv',0)


# In[43]:


ds.shape


# In[44]:


ds.head()


# In[45]:


pat(ds)
make_test(ds)


# In[46]:


test_sentences = ds.texto.to_numpy()
test_labels = ds.valor.to_numpy()


# In[47]:


test_sequences = tokenizer.texts_to_sequences(test_sentences)


# In[48]:


test_padded = pad_sequences(test_sequences, maxlen=max_length, padding = "post",truncating = "post")


# In[49]:


predictions_t = model.predict(test_padded)
predictions_t = [np.argmax(element) for element in predictions_t]
print(predictions_t)

print(precision(test_labels,predictions_t))


# In[ ]:





# In[ ]:





# In[ ]:




