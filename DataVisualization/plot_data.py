#%%

from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from collections import Counter
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

#%%

train = pd.read_csv('data/clean/train.csv')
cv = pd.read_csv('data/clean/cv.csv')
test = pd.read_csv('data/clean/test.csv')

all_data = train.append(cv).append(test)
all_data.created_utc = pd.to_datetime(all_data.created_utc)

#%%

plt.figure(figsize=(5,5))
ax = sns.countplot(x='label',  data= all_data)
ax.set(title = "Distribution of Classes", xlabel="Sarcasm Status", ylabel = "Total Count")
plt.show()

#%%

sns.boxplot(x= all_data.loc[all_data['label'] == 1, 'comment'].str.len()).set(title = 'Len of Sarcastic Comments', xlabel = 'Length')
sns.despine(offset=10, trim=True)
plt.show()

#%%

sns.boxplot(x= all_data.loc[all_data['label'] == 0, 'comment'].str.len()).set(title = 'Len of Neutral Comments', xlabel = 'Length')
sns.despine(offset=10, trim=True)
plt.show()

#%%

all_data['log_comment'] = all_data['comment'].apply(lambda text: np.log1p(len(str(text))))
all_data[all_data['label']==1]['log_comment'].hist(alpha=0.6,label='Sarcastic', color = 'blue')
all_data[all_data['label']==0]['log_comment'].hist(alpha=0.6,label='Non-Sarcastic', color = 'red')
plt.legend()
plt.title('Natural Log Length of Comments')
plt.show()

#%%

wordcloud = WordCloud(background_color='black', stopwords = STOPWORDS,
                      max_words = 200, max_font_size = 100,
                      random_state = 17, width=600, height=400)

plt.figure(figsize=(12, 12))
wordcloud.generate(str(all_data.loc[all_data['label'] == 1, 'comment']))
plt.grid(b= False)
plt.imshow(wordcloud);


#%%

wordcloud = WordCloud(background_color='black', stopwords = STOPWORDS,
                      max_words = 200, max_font_size = 100,
                      random_state = 17, width=600, height=400)

plt.figure(figsize=(12, 12))
wordcloud.generate(str(all_data.loc[all_data['label'] == 0, 'comment']))
plt.grid(b= False)
plt.imshow(wordcloud);

#%%

sarcasm_score = np.array(all_data.loc[all_data['label'] == 1]['score'])
neutral_score = np.array(all_data.loc[all_data['label'] == 0]['score'])

labels = ['Sarcastic Score', 'Neutral Score']
sizes = [3235069, 3725113]

plt.rcParams.update({'font.size': 14})
fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = ["blue","yellow"], labels=labels, autopct='%1.1f%%', startangle=30)
ax1.set_title("Scores of Subreddits")

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

ax1.axis('equal')
plt.tight_layout()
plt.show()

#%%

# Getting the top 5 popular subreddits
print(all_data['subreddit'].value_counts()[:5])

top_subreddits =['AskReddit', 'politics', 'worldnews', 'leagueoflegends', 'pcmasterrace']

subreddit = pd.DataFrame()
subreddit['subreddit'] = top_subreddits
subreddit['sarcastic'] = np.nan
subreddit['natural'] = np.nan
subreddit['total'] = np.nan

# Calculating the count of Sarcastic and Natural comments for the top 5 subreddits
for i in range(len(top_subreddits)):
    temp = all_data.loc[all_data['subreddit'] == subreddit.subreddit.iloc[i]]
    length = len(temp)
    count_sarcastic = len(temp.loc[temp['label'] == 1])
    subreddit.sarcastic.iloc[i] = count_sarcastic
    subreddit.natural.iloc[i] = length - count_sarcastic
    subreddit.total.iloc[i] = length


#%%

# Feature Engineering- Extracting the day of a week
all_data['created_utc'] = pd.to_datetime(all_data['created_utc'], format = '%d/%m/%Y %H:%M:%S')
all_data['Day of Week'] = all_data['created_utc'].dt.day_name()

# Visualization of Column- label
plt.figure(figsize=(12, 12))
ax = sns.countplot(x='Day of Week',  data= all_data.loc[all_data['label']==1])
ax.set(title = "Count of sarcastic comments per day", xlabel="Days of the week", ylabel = "Total Count")
total = float(len(all_data ))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 7,
            '{:1.1f}%'.format((height/total)*100*2),
            ha="center")
plt.show()

#%%

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

sarcasm_subjectivity = all_data.loc[all_data['label'] == 1, 'comment'].astype(str).apply(get_subjectivity)
non_sarcasm_subjectivity = all_data.loc[all_data['label'] == 0, 'comment'].astype(str).apply(get_subjectivity)

print(f"Subjectivity score for sarcastic comments:{sarcasm_subjectivity.sum()}")
print(f"Subjectivity score for for non sarcastic comments: {non_sarcasm_subjectivity.sum()}")

#%%

def get_count_pos(text):
    tokens = nltk.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    the_count = Counter(tag[0][0] for _, tag in pos if tag[0][0].startswith('V') or tag[0][0].startswith('N') or tag[0][0].startswith('J') or tag[0][0].startswith('P'))
    return the_count

sarcasm_pos = all_data.loc[all_data['label'] == 1, 'comment'].astype(str).apply(get_count_pos)
non_sarcasm_pos = all_data.loc[all_data['label'] == 0, 'comment'].astype(str).apply(get_count_pos)

sarcasm_pos = reduce((lambda x, y: x + y), sarcasm_pos)
non_sarcasm_pos = reduce((lambda x, y: x + y), non_sarcasm_pos)

print(f"Part of speech counter for sarcastic comments:{sarcasm_pos}")
print(f"Part of speech counter for non sarcastic comments: {non_sarcasm_pos}")

#%%


