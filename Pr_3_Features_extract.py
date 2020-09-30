#!/usr/bin/env python
# coding: utf-8

# In[1]:


A = [' M.S. Dhoni is running towards the #catch.', 'M.S. Dhoni is better than Gilchrist','Gilchrist is better than M.S. Dhoni','M.S. Dhoni is the no 1 football player','M.S. Dhoni M.S. Dhoni GilchristGilchrist']


# In[2]:


import pandas as pd
data = pd.DataFrame({'tweet_text':A})


# In[ ]:


data['tweet_text']


# In[3]:


#Number of Words
data['word_count'] = data['tweet_text'].apply(lambda x: len(str(x).split(" ")))
data[['tweet_text','word_count']].head()


# In[4]:


#Number of characters
data['char_count'] = data['tweet_text'].str.len() ## this also includes spaces
data[['tweet_text','char_count']].head()


# In[5]:


#Average Word Length
#Number of characters(without space count)/Total number of words
def avg_word(sentence):
  words = sentence.split()
  print(words)
  print(len(words))
  print(sum(len(word) for word in words))
  return (sum(len(word) for word in words)/len(words))

data['avg_word'] = data['tweet_text'].apply(lambda x: avg_word(x))
data[['tweet_text','avg_word']].head()


# In[6]:


#Number of stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['stopwords'] = data['tweet_text'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['tweet_text','stopwords']].head()


# In[7]:


#Number of special characters
data['hastags'] = data['tweet_text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
data[['tweet_text','hastags']].head()


# In[8]:


#Number of numerics
data['numerics'] = data['tweet_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['tweet_text','numerics']].head()


# In[9]:


#Number of Uppercase words
data['upper'] = data['tweet_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
data[['tweet_text','upper']].head()


# In[11]:


pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence

from textblob import TextBlob, Word, Blobber
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
                print(ppo, tup)
    except:
        pass
    return cnt

data['noun_count'] = data['tweet_text'].apply(lambda x: check_pos_tag(x, 'noun'))
data['verb_count'] = data['tweet_text'].apply(lambda x: check_pos_tag(x, 'verb'))
data['adj_count'] = data['tweet_text'].apply(lambda x: check_pos_tag(x, 'adj'))
data['adv_count'] = data['tweet_text'].apply(lambda x: check_pos_tag(x, 'adv'))
data['pron_count'] = data['tweet_text'].apply(lambda x: check_pos_tag(x, 'pron'))
data[['tweet_text','noun_count','verb_count','adj_count', 'adv_count', 'pron_count' ]].head()


# In[12]:


data.head()


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[14]:


cv=CountVectorizer()
A_vec = cv.fit_transform(A)
print(A_vec.toarray())


# In[15]:


tv=TfidfVectorizer()
t_vec = tv.fit_transform(A)
print(t_vec.toarray())


# In[16]:


feature_names = tv.get_feature_names()

dense = t_vec.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)


# In[17]:


feature_names


# In[18]:


df_c =pd.concat([df,data], axis=1)
df_c.head()


# In[ ]:




