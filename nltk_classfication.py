#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:12:35 2020

@author: peterfang
"""
import spacy
import string
from spacy.lang.en import English
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
import en_core_web_sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def classfication(path):
    sample_data=pd.read_csv('research_paper.csv')
    
    ## assign dummy numbers to paper categories
    lookup_dic={"ISCAS":1,"INFOCOM":2, "VLDB":3,"WWW":4,"SIGGRAPH":5 }   
    sample_data['Conference_re']=sample_data['Conference'].apply(lambda x: lookup_dic.get(x, "value"))
    
    # Create the list of punctuation marks
    punctuations = string.punctuation
    
    # Create the list of stopwords
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    
    # Load English tokenizer, tagger, parser
    parser = English()
    
    def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
        mytokens = parser(sentence)
        # Lemmatizing each token and converting each token into lowercase
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

        # Removing stop words
        mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    
        # return preprocessed list of tokens
        return mytokens 

    #####bag of words and efidf
    bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
    
    #tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
    X, ylabels = sample_data['Title'],sample_data['Conference_re'] # the features we want to analyze
    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)
    
    classifier = LogisticRegression()
    
    # Create pipeline using Bag of Words
    pipe = Pipeline([('vectorizer', bow_vector),
                     ('classifier', classifier)])
    #model
    pipe.fit(X_train,y_train)
    predicted = pipe.predict(X_test)
    
    frame=pd.DataFrame(data={'acutal':y_test, 'pred':predicted, 'paper':X_test})
    # validation    
    mse=mean_squared_error(frame['acutal'], frame['pred'])
    R2=r2_score(frame['acutal'], frame['pred'])
    print ("mse is {}".format(mse))
    print ("R2 is {}".format(R2))
    
    frame.to_csv(path+'research_paper_prediction.csv')
    
if __name__== "__main__":
# model generation
   path="/Users/*/work/python/" 
   classfication(path)
        
######countvector from sklearn
# =============================================================================
# from sklearn.feature_extraction.text import CountVectorizer
# corpus = [
#      'This is the first document.',
#      'This document is the second document.',
#      'And this is the third one.',
#      'Is this the first document?',
# ]
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
# ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
# print(X.toarray())
# [[0 1 1 1 0 0 1 0 1]
#  [0 2 0 1 0 1 1 0 1]
#  [1 0 0 1 1 0 1 1 1]
#  [0 1 1 1 0 0 1 0 1]]
# vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
# X2 = vectorizer2.fit_transform(corpus)
# print(vectorizer2.get_feature_names())
# ['and this', 'document is', 'first document', 'is the', 'is this',
# 'second document', 'the first', 'the second', 'the third', 'third one',
#  'this document', 'this is', 'this the']
# print(X2.toarray())
#  [[0 0 1 1 0 0 1 0 0 0 0 1 0]
#  [0 1 0 1 0 1 0 1 0 0 1 0 0]
#  [1 0 0 1 0 0 0 0 1 1 0 1 0]
#  [0 0 1 0 1 0 1 0 0 0 0 0 1]]
# =============================================================================
