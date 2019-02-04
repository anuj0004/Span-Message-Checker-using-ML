import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_table('SMSSpamCollection', header = None , encoding='utf-8')

print(df.info())
print(df.head())

# to check the unique value in the coloum 

classes = df[0]
print(classes.value_counts())

#convert the catorical data into numirical daata or lable encoding

from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()

Y=encoder.fit_transform(classes)

print(Y[:10])
print(classes[:10])

text_message= df[1]
print(text_message[:10])

processed= text_message.str.replace(r'^\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}$','emailaddr')

processed= processed.str.replace(r'http\://[a-zA-Z0-9\-\.]+\.{a-zA-Z}{2,3}(/\S*)?$','webaddress')
processed= text_message.str.replace(r'£|\$','moneysymb')
processed= text_message.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr')

processed= text_message.str.replace(r'\d+(\.\d+)?','numbr')

processed= text_message.str.replace(r'[^\w\d\s]',' ')

processed= text_message.str.replace(r'\s+',' ')

processed= text_message.str.replace(r'^\s+|\s+?$','')

processed = processed.str.lower()
print(processed)

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
ps= nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(ps.stem() for term in x.split()))
print(processed)
from nltk.tokenize import word_tokenize

all_words =[]
for message in processed:
    words =word_tokenize(message)
    for w in words:
        all_words.append(w)
        
'''all_words = nltk.FreqDist(all_words)
print('Number of Words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.mostcommon(15)))'''




