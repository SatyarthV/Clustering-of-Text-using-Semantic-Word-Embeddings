#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:40:44 2017

@author: satyarthvaidya
"""
import pandas as pd
import gensim
import nltk.data
from nltk.corpus import stopwords
import re
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.decomposition import PCA

#READING THE DATA
path = r"/home/satyarthvaidya/Satyarth's Stuff/News Dataset/demo.csv"
data = pd.read_csv(path, header=None)

data['Articles'] = data

sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
punctuation_tokens = [ '\n','/',"'—" ,' #'  ,'”',  '·',' . ','’', '‘','%','.','#','@' ,"'—",'|','«','» |','|','»','.', '..', '...', ',', ';', ':', '(', ')', '"', '“',',','„','\'', '[', ']', '{', '}', '?', '!', '-', u'–', '+', '*', '--', '\'\'', '``']
punctuation = "|?.!/;:()&+„.“|0-9'— ·#\r\n\r\n\r\n\xa0-\xa0\r\n\r\n/"

stop_words = stopwords.words('english') 
data['Articles_processed'] = data['Articles'].apply(lambda x: x.lower())
data['Articles_processed'] = data['Articles'].str.replace(r'\d+', '')

data['Tokenized'] = data['Articles_processed'].apply(nltk.word_tokenize)
data['Tokenized'] = data['Tokenized'].apply(lambda x : [item for item in x if item not in punctuation_tokens])
data['Tokenized'] = data['Tokenized'].apply(lambda x : [ re.sub('[' + punctuation + ']', '', item) for item in x])
data['Tokenized'] = data['Tokenized'].apply(lambda x : [item for item in x if item not in stop_words])

sentences  = data.Tokenized.tolist()

#Doc2vec model training
sentences_ = [gensim.models.doc2vec.LabeledSentence(words_lis, ['SENT_%d'%index]) \
              for index,words_lis in enumerate(list(data['Tokenized'].ravel()))]

d2v_model = gensim.models.Doc2Vec(size=100,alpha=0.025,min_alpha=0.025,window=8,min_count=5,seed=1,workers=4)
d2v_model.build_vocab(sentences_)

for epoch in range(10):
    print('Epoch %d'%epoch)
    random.shuffle(sentences_)
    d2v_model.train(sentences_ , total_examples = d2v_model.corpus_count, epochs = d2v_model.iter)
#    d2v_model.train(sentences_)
    d2v_model.alpha-=0.0002
    d2v_model.min_alpha = d2v_model.alpha

data['dvecs'] = [d2v_model.docvecs['SENT_%d'%i] for i in range(data.shape[0])]
new_data = data['dvecs'] 
transformed_data = np.vstack(new_data)
#for j in range(2,11):
reduced_data = PCA(n_components=3).fit_transform(transformed_data)

#for i in range(11,21):
clusters=5
km = KMeans(n_clusters=clusters , random_state  = 0 ,  init='k-means++')
km.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     
# point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
z_min, z_max = reduced_data[:, 2].min() - 1, reduced_data[:, 2].max() + 1
xx, yy , zz= np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = km.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max(), zz.min(), zz.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:2], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = km.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the Document Vectors (PCA-reduced data) ' )
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

#saving and reloading the model
import pickle

pickle.dump(model1,open('model1.p','wb'))

d2v_model = pickle.load(open('model1.p','rb'))

