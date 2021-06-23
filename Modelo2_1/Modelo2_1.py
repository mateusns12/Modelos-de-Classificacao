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


ds = pd.read_csv("train.csv",encoding="utf-8")


# In[3]:


df = ds.sample(frac=1)
df['texto'] = df['texto'].apply(str)


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.texto


# In[7]:


print((df.valor==1).sum())#eletrica
print((df.valor==0).sum())#direito


# In[8]:


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


# In[9]:


pattern = re.compile(r"https?//(\S+|www)\.\S+")
for t in df.texto:
    matches = pattern.findall(t)
    for match in  matches:
        print(t)
        print(match)
        print(pattern.sub(r"",t))
        
    if len(matches)> 0:
        break


# In[10]:


df["texto"] = df.texto.map(remove_URL)
df["texto"] = df.texto.map(remove_punct)
df["texto"] = df.texto.map(remove_hifen)


# In[11]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words("portuguese"))ss

def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


# In[12]:


stop


# In[13]:


df["texto"] = df.texto.map(remove_stopwords)


# In[14]:


df.texto


# In[15]:


df.texto[201]


# In[16]:


from collections import Counter

def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count

counter = counter_word(df.texto)


# In[17]:


len(counter)


# In[18]:


counter.most_common(5)


# In[19]:


num_unique_words = len(counter)


# In[20]:


train_size = int(df.shape[0]*0.8)

train_df = df[:train_size]
val_df = df[train_size:]


# In[21]:


print(len(train_df))
print(len(val_df))


# In[22]:


train_sentences = train_df.texto.to_numpy()
train_labels = train_df.valor.to_numpy()

val_sentences = val_df.texto.to_numpy()
val_labels = val_df.valor.to_numpy()


# In[23]:


train_sentences.shape, val_sentences.shape


# In[24]:


from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = num_unique_words,oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)


# In[25]:


word_index = tokenizer.word_index


# In[26]:


word_index


# In[27]:


train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)


# In[28]:


print(train_sentences[12],'\n')
print(train_sequences[12])


# In[29]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 100

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding = "post",truncating = "post")
val_padded = pad_sequences(val_sequences, maxlen = max_length, padding = "post", truncating = "post")
train_padded.shape, val_padded.shape


# In[30]:


train_padded[10]


# In[31]:


print(train_sentences[10])
print(train_sequences[10])
print(train_padded[10])


# In[32]:


reverse_word_index = dict([(idx,word) for (word, idx) in word_index.items()])


# In[33]:


reverse_word_index


# In[34]:


def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])


# In[35]:


decoded_text = decode(train_sequences[10])

print(train_sequences[10])
print(decoded_text)


# In[82]:


from tensorflow.keras import layers

model = keras.models.Sequential()
model.add(layers.Embedding(num_unique_words, 32, input_length = max_length))

model.add(layers.LSTM(256,dropout = 0.1))
#model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(24, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))

model.summary()


# In[83]:


loss = keras.losses.BinaryCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer = optim, metrics = metrics)


# In[84]:


import time
start = time.perf_counter()
model.fit(train_padded,train_labels, epochs = 20, validation_data=(val_padded,val_labels), verbose=2)
finish = time.perf_counter()
print(f'\nFinished in {round(finish-start, 2)} second(s)')


# In[85]:


predictions = model.predict(val_padded)
#print(predictions)
predictions = [1 if p > 0.5 else 0 for p in predictions]


# In[88]:


def precision():
    counter = len(val_labels)
    list_c = [i for i,j in zip(predictions,val_labels) if i == j]
    return counter,len(list_c)
t,p = precision()
print(p*100/t)


# In[89]:


index = 7
print(val_sentences[index])
print(val_padded[index])
print("Label: ",val_labels[index])
print("Resultado: ",predictions[index])
print(val_labels)
print(predictions)


# In[42]:


df_s = pd.read_csv("eval.csv")
df_t = df_s.sample(frac=1)
df_t['texto'] = df_t['texto'].apply(str)


# In[43]:


df_t.shape


# In[44]:


df_t.head()


# In[45]:


def make_test():
    df_t["texto"] = df.texto.map(remove_URL)
    df_t["texto"] = df.texto.map(remove_punct)
    df_t["texto"] = df.texto.map(remove_hifen)
    df_t["texto"] = df.texto.map(remove_stopwords)


# In[46]:


def pat():
    for t in df_t.texto:
        matches = pattern.findall(t)
        for match in  matches:
            print(t)
            print(match)
            print(pattern.sub(r"",t))        
        if len(matches)> 0:
            break


# In[47]:


pat()
make_test()


# In[48]:


df_t.texto


# In[49]:


test_sentences = df_t.texto.to_numpy()
test_labels = df_t.valor.to_numpy()


# In[79]:


test_sequences = tokenizer.texts_to_sequences(test_sentences)
word_index_test = tokenizer.word_index


# In[51]:


#print(test_sentences)
print(test_sequences[1])


# In[52]:


test_padded = pad_sequences(test_sequences, maxlen=max_length, padding = "post",truncating = "post")


# In[80]:


#print(test_padded)


# In[90]:


predictions_t = model.predict(test_padded)
print(predictions_t)
predictions_t = [1 if p > 0.5 else 0 for p in predictions_t]


# In[91]:


index = 6
#print(test_sentences)#[index])

print(test_labels)#[index])
print(predictions_t)#[index])

def precision_t():
    counter = len(test_labels)
    list_c = [i for i,j in zip(predictions_t,test_labels) if i == j]
    return counter,len(list_c)
t,p = precision_t()
print(p*100/t," %")


# In[56]:


decoded_test = decode(test_sequences[7])


# In[57]:


print(decoded_test)


# In[58]:


#model.save_weights('m_salvo/')


# In[59]:


#model.save('modelo_completo/')


# In[60]:


teste_2 = "Este trabalho descreve o projeto de um carregador de baterias tipo chumbo-ácido controlado por um microcontrolador, utilizando os princípios e técnicas da eletrônica de potência. O sistema será capaz de carregar um banco de baterias composto por até seis baterias dispostas em série, controlando o processo de carga para garantir a integridade do sistema e otimizar a vida útil das mesmas, utilizando métodos inteligentes para o processo de carga. O banco de baterias pertence a um projeto em andamento que trata do desenvolvimento de um veículo náutico autônomo."


# In[61]:


teste_2 = remove_URL(teste_2)
teste_2 = remove_punct(teste_2)
teste_2 = remove_hifen(teste_2)
teste_2 = remove_stopwords(teste_2)


# In[62]:


teste_2


# In[63]:


texto = tokenizer.texts_to_sequences([teste_2])


# In[64]:


texto


# In[65]:


import numpy as np
ar = np.array(texto)


# In[66]:


ar


# In[67]:


predictions_t2 = model.predict(ar)
print(predictions_t2)
predictions_t2 = [1 if p > 0.5 else 0 for p in predictions_t2]

print(predictions_t2)


# In[68]:


teste_3 = "Este trabalho de conclusão de curso teve como objetivo simular e analisar a cascata energética do rio tietê que pertencem a aes-tietê. As usinas hidrelétricas que foram estudas são: barra bonita, bariri, ibitinga, promissão e nova avanhandava. Para simulação foi usado o software mike basin 2000. E a satisfação, que pode ser definida como a probabilidade de atendimento das demandas totais do sistema foi usada para análise. Foram simulados dois cenários: o primeiro representando um caso atual, onde foram escolhidos os anos de 1998 à 2007 e segundo um caso crítico. Este último foram escolhido as dez piores médias anuais entre os anos de 1931 à 2007. Em relação ao cenário 1, que representa os últimos dez anos de vazões naturais, observou uma grande satisfação em todas as usinas hidrelétricas, mesmo tendo alguns períodos de seca. Chegou até verter água na uhe de nova avanhandava. O cenário 2 demonstra como o sistema reagiria em caso de uma seca prolongada de dez anos. Como foi visto, diminuiria bastante a produção de energia, mas não chegaria a zero"


# In[69]:


teste_3 = remove_URL(teste_3)
teste_3 = remove_punct(teste_3)
teste_3 = remove_hifen(teste_3)
teste_3 = remove_stopwords(teste_3)


# In[70]:


texto_3 = tokenizer.texts_to_sequences([teste_3])


# In[71]:


predictions_t2 = model.predict(np.array(texto_3))
print(predictions_t2)
predictions_t2 = [1 if p > 0.5 else 0 for p in predictions_t2]

print(predictions_t2)


# In[72]:


teste_4 = "A relativização da coisa julgada atualmente ganha grande notoriedade no sistema jurídico brasileiro, o que se revela, de certo modo, um meio de buscar e atingir uma justiça plena, ou seja, busca-se garantir maior segurança e evitar prejuízos às relações jurídico-sociais. Sendo assim, não é de se espantar que a relativização da coisa julgada tenha ganhado destaque nos últimos tempos, visto que as relações tendem a se tornar cada vez mais complexas e difíceis, o acerto se faz algo nem sempre tão exato. O artigo 467 do CPC, em linhas gerais, diz que a coisa julgada torna indiscutível e imutável a decisão, não mais sujeita a recurso ordinário ou extraordinário. O texto faz referência a chamada coisa julgada material, a qual tem seus efeitos além processo, ou seja, extrapolam o âmbito processual, bem como impede de rediscuti-lo e vincula o magistrado à sua decisão. A coisa julgada material é garantia constitucional (art. 5º, XXXVI, CF) e protegida em nível de cláusula pétrea, logo inerente ao Estado democrático de direito e ao acesso ao Judiciário, portanto relativizar a coisa julgada seria apenas possível através de uma Assembléia Constituinte Originária. Todavia a dinâmica da sociedade fez necessária, em alguns casos, sua relativização, já que um dos principais ideais é buscar sempre atingir a justiça, mesmo que para isso haja detrimento da segurança jurídica..."


# In[73]:


teste_4 = remove_URL(teste_4)
teste_4 = remove_punct(teste_4)
teste_4 = remove_hifen(teste_4)
teste_4 = remove_stopwords(teste_4)


# In[74]:


texto_4 = tokenizer.texts_to_sequences([teste_4])


# In[75]:


predictions_t5 = model.predict(np.array(texto_3))
print(predictions_t5)
predictions_t5 = [1 if p > 0.5 else 0 for p in predictions_t5]

print(predictions_t5)


# In[ ]:




