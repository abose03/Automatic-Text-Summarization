#!/usr/bin/env python
# coding: utf-8

# # Corrupted Data Generation

'''reading the dataset'''

import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
train_df = pd.read_csv('/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv')
train_df.head(2)


# In[ ]:


#loading libraries
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm") #loading spacy model for english pipeline 

# function to generate corrupted data taking argument as original summaries of train dataset

def show_ents(doc1): 

    df=pd.DataFrame()
    l=[]
    l1=[]

    org.append(doc1)
    doc = nlp(doc1) #creating nlp object

    if doc.ents:  # finding all entities & its label and storing in list 'l1' and 'l' respectively 
        for ent in doc.ents: 
            l.append(ent.label_)
            l1.append(ent.text)

    else: print('No named entities found.')


    # counting number of entities , entity label wise and storing them in df1
    df['ent_lab']=l
    df['ent']=l1
    df1 = df['ent_lab'].value_counts().rename_axis('unique_values').reset_index(name='counts')

    #if entity label count is >1 replace the 1st two entities for that entity label
    if (df['ent_lab'].value_counts()>1).any():
        text=doc1[:]
        x=df1.unique_values[df1.counts>1]
        for i in range(0,len(x)):
            y=df.ent.loc[df.ent_lab==x[i]]
            y=y.reset_index(drop=True)
            text=text.replace(y[0],'mm').replace(y[1],y[0]).replace('mm',y[1])  #replacement
        data.append(text) #list to store corrupted summary

    #if entity label count is <1 replace the VERB with negation eg: used => did not use
    else:
        text=doc1[:]
        doc_dep = nlp(text)
        for i in range(len(doc_dep)):
            token = doc_dep[i]
            if token.tag_ in ['VBP','VBZ','VBD']: #finding verbs
                text = text.replace(token.text, 'did not'+ ' ' +token.text) #replacement with negation
                text = text.replace(token.text, token._.inflect("VB")) #replacement with negation
        data.append(text) #list to store corrupted summary


# In[ ]:


data=[]
org=[]
# train_df = pd.read_csv('/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv')

#genearting corrupted data for 100 summaries
for i in range (0,100):
    text=train_df['highlights'][i]

    #calling the above function with the original summary as input argument
    show_ents(text)
    
data1={'original':org,'noisy':data} 
df=pd.DataFrame(data1)  # dataframe has both original summary & corrupted summary

#saving to csv
df.to_csv('noisy.csv')

