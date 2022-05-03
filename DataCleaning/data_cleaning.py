#%%

import re
import warnings

import pandas as pd
import seaborn as sns
warnings.filterwarnings('ignore')

#%% md

# Data

#%% md

## Loading data

#%%

train = pd.read_csv('data/raw/train.csv')
cv = pd.read_csv('data/raw/cv.csv')
test = pd.read_csv('data/raw/test.csv')

#%%

train.head()

#%%

train.shape

#%% md

We've loaded our balanced data set with 800K points.

#%%

train.info()

#%% md

## Exploratory Data Analysis (EDA)

#%%

train['label'].value_counts()

#%%

sns.countplot(x='label', data=train)

#%%

import neattext as nt

def removeWordsWithNumbers(text):
    return re.sub(r'\S*\d\S*', '', text).strip()

def remove_emoticons(text):
    emoticon_pattern = re.compile(u"\U0001F600-\U0001F64F", flags=re.UNICODE)
    return emoticon_pattern.sub(r'', text)

#%%

def clean_data(content):
    content = str(content)
    content = remove_emoticons(content)
    content = removeWordsWithNumbers(content)
    docx = nt.TextFrame(text=content)
    docx.remove_emojis()
    docx.remove_html_tags()
    docx.remove_puncts()
    docx.remove_urls()
    docx.remove_stopwords(lang='en')
    docx.remove_special_characters()
    docx.fix_contractions()
    docx.remove_numbers()

    return ' '.join(content.lower().strip().split())

#%%

cleaned_train = train.dropna(how='any', axis=0)
cleaned_train['comment'] = cleaned_train['comment'].apply(clean_data)
cleaned_train['author'] = cleaned_train['author'].apply(lambda x: x.strip())

cleaned_cv = cv.dropna(how='any', axis=0)
cleaned_cv['comment'] = cleaned_cv['comment'].apply(clean_data)
cleaned_cv['author'] = cleaned_cv['author'].apply(lambda x: x.strip())

cleaned_test = test.dropna(how='any', axis=0)
cleaned_test['comment'] = cleaned_test['comment'].apply(clean_data)
cleaned_test['author'] = cleaned_test['author'].apply(lambda x: x.strip())

#%%

#testing
train['comment'].iloc[66]

#%%

#testing
cleaned_train['comment'].iloc[66]

#%%

cleaned_train['comment'] = cleaned_train['comment'].astype(str)
cleaned_cv['comment'] = cleaned_cv['comment'].astype(str)
cleaned_test['comment'] = cleaned_test['comment'].astype(str)

cleaned_train['author'] = cleaned_train['author'].astype(str)
cleaned_cv['author'] = cleaned_cv['author'].astype(str)
cleaned_test['author'] = cleaned_test['author'].astype(str)

#%%

cleaned_train.to_csv('data/clean/train.csv', index=None)
cleaned_cv.to_csv('data/clean/cv.csv', index=None)
cleaned_test.to_csv('data/clean/test.csv', index=None)

#%%



#%%


