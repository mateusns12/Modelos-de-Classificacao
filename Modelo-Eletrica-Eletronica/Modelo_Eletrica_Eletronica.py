#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np


# In[2]:


def get_data(nome_arquivo,shuffle):
    ds = pd.read_csv(nome_arquivo,encoding="utf-8")
    if shuffle:
        ds = ds.sample(frac=1)
    ds['texto'] = ds['texto'].apply(str)
    return ds


# In[3]:


df = get_data("train.csv",1)


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.texto


# In[7]:


print((df.valor==1).sum())#eletronica
print((df.valor==0).sum())#eletrica


# In[8]:


import re
import string
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

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


# In[9]:


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
pat(df)            


# In[10]:


def make_test(df_t):
    df_t["texto"] = df_t.texto.map(remove_URL)
    df_t["texto"] = df_t.texto.map(remove_punct)
    df_t["texto"] = df_t.texto.map(remove_hifen)
    df_t["texto"] = df_t.texto.map(remove_numbers)
    df_t["texto"] = df_t.texto.map(remove_stopwords)

make_test(df)


# In[11]:


from collections import Counter

def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count

counter = counter_word(df.texto)


# In[12]:


len(counter)


# In[13]:


counter.most_common(5)


# In[14]:


num_unique_words = len(counter)


# In[15]:


def data_split(df,size):
    train_size = int(df.shape[0]*size)
    train_df = df[:train_size]
    val_df = df[train_size:]
    return train_df, val_df

train_df, val_df = data_split(df,0.8)


# In[16]:


print(len(train_df))
print(len(val_df))


# In[17]:


def data_to_numpy(df):    
    train_sentences = train_df.texto.to_numpy()
    train_labels = train_df.valor.to_numpy()
    val_sentences = val_df.texto.to_numpy()
    val_labels = val_df.valor.to_numpy()
    return train_sentences, train_labels, val_sentences, val_labels

train_sentences, train_labels, val_sentences, val_labels = data_to_numpy(df)


# In[18]:


train_sentences.shape, val_sentences.shape


# In[19]:


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = num_unique_words,oov_token="<OOV>")

def tokenization(df):      
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    return train_sequences, val_sequences, word_index

train_sequences, val_sequences, word_index = tokenization(df)


# In[20]:


#word_index = tokenizer.word_index


# In[21]:


#train_sequences = tokenizer.texts_to_sequences(train_sentences)
#val_sequences = tokenizer.texts_to_sequences(val_sentences)


# In[22]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 600

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding = "post",truncating = "post")
val_padded = pad_sequences(val_sequences, maxlen = max_length, padding = "post", truncating = "post")
train_padded.shape, val_padded.shape


# In[23]:


reverse_word_index = dict([(idx,word) for (word, idx) in word_index.items()])


# In[24]:


def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])


# In[25]:


decoded_text = decode(train_sequences[10])

print(train_sequences[10])
print(decoded_text)


# In[26]:


from tensorflow.keras import layers

model = keras.models.Sequential()
model.add(layers.Embedding(num_unique_words, 32, input_length = max_length))

#model.add(layers.LSTM(256,dropout = 0.1))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(128, activation = "relu"))
model.add(layers.Dense(128, activation = "relu"))
model.add(layers.Dense(24, activation = "relu"))
model.add(layers.Dense(2, activation = "softmax"))


model.summary()


# In[27]:


#loss = keras.losses.BinaryCrossentropy(from_logits=False)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer = optim, metrics = metrics)


# In[28]:


import time
start = time.perf_counter()
model.fit(train_padded,train_labels, epochs = 20, validation_data=(val_padded,val_labels), verbose=2)
finish = time.perf_counter()
print(f'\nFinished in {round(finish-start, 2)} second(s)')


# In[29]:


predictions = model.predict(val_padded)
print([np.argmax(element) for element in predictions])

#predictions = [1 if p > 0.5 else 0 for p in predictions]


# In[30]:


index = 7
#print(val_sentences[index])
#print(val_padded[index],'\n')
print("Label: ",val_labels[index])
print("Resultado: ",predictions[index],'\n')
print(val_labels,'\n')
print(predictions)


# In[31]:


df_t = get_data("eval.csv",0)


# In[32]:


df_t.shape


# In[33]:


df_t.head()


# In[34]:


pat(df_t)
make_test(df_t)


# In[35]:


df_t.texto


# In[36]:


test_sentences = df_t.texto.to_numpy()
test_labels = df_t.valor.to_numpy()


# In[37]:


test_sequences = tokenizer.texts_to_sequences(test_sentences)


# In[38]:


test_padded = pad_sequences(test_sequences, maxlen=max_length, padding = "post",truncating = "post")


# In[39]:


predictions_t = model.predict(test_padded)
print(predictions_t)
#predictions_t = [1 if p > 0.5 else 0 for p in predictions_t]


# In[40]:


index = 6
print(test_sentences[2])#[index])

print(test_labels)#[index])
#print(predictions_t)#[index])
predictions_t = [np.argmax(element) for element in predictions_t]
print(predictions_t)
def precision_t():
    counter = len(test_labels)
    list_c = [i for i,j in zip(predictions_t,test_labels) if i == j]
    return len(list_c)/counter*100

print(precision_t(),"%")


# In[41]:


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


# In[42]:


teste_1 = "O objetivo deste trabalho é propor um algoritmo para realizar a identificação de padrões na vocalização suína, visando determinar o nível do bem-estar do animal. Tal análise foi proposta uma vez que o bem-estar animal é um assunto cada vez mais abordado no mundo todo, principalmente quando os animais são criados para o abate. Dessa forma, a criação de um método em que haja o mínimo de contato com os animais se faz importante, evitando que tal contato altere o comportamento do animal e, conseqüentemente, o resultado da análise de seu bem-estar. Por essas características, foi proposto um método de análise dos sons emitidos pelos suínos com base na utilização de uma Rede Neural Artificial do tipo Radial Basis Function, a qual possui como elementos de treinamento e operação um conjunto de características extraídas através da Transformada Discreta Wavelet de sinais sonoros pré-gravados. As características obtidas dos sinais foram as energias das bandas críticas relativas à Escala Bark e a diferença entre as energias das bandas adjacentes, além dimensão fractal do sinal. Através desse método foram analisados dois tipos de sinais sonoros: a vocalização de leitões saudáveis e de leitões acometidos por uma doença chamada Artrite Traumática; e a vocalização de suínos adultos em situações de conforto e desconforto. Os resultados demonstram que a análise proposta atingiu bons patamares de acerto na determinação do bem-estar do animal"
#eletronica


# In[43]:


teste_1 = tokenizer.texts_to_sequences([prepare(teste_1)])


# In[44]:


predict(teste_1)


# In[45]:


teste_2 = "Neste projeto de formatura seria dada continuidade ao trabalho de iniciação científica, que consistiu na análise de métodos de coordenação de robôs móveis, ou seja métodos de controle de um conjunto de robôs para que esses assumam uma determinada formação com direção e distância entre eles pré-definidas. Sendo estudada primeiramente a cinemática do robô móvel e logo em seguida foi feita a análise de uma formação simples de líder-seguidor, considerando também a alternância da liderança entre os robôs. A partir desse ponto seria feita a implementação da simulação do algoritmo de controle de 6 robôs móveis em formação utilizando controle por grafos e sistemas lineares sujeitos a saltos markovianos"
#eletronica


# In[46]:


teste_2 = tokenizer.texts_to_sequences([prepare(teste_2)])


# In[47]:


predict(teste_2)


# In[ ]:





# In[ ]:




