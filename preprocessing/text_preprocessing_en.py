import emoji
import tqdm
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, sent_tokenize
from nltk.stem import PorterStemmer
import nltk

#텍스트 집중 전처리를 위해 이모지는 별도 코드로 변환
reviews = pd.read_csv('/content/drive/MyDrive/dd/reviews.csv')
for row in tqdm.tqdm(range(reviews.shape[0])):
  reviews['review'][row] = emoji.demojize(reviews['review'][row])
  if type(reviews['title'][row]) == 'str':
    reviews['title'][row] = emoji.demojize(reviews['title'][row])

for row in tqdm.tqdm(range(reviews.shape[0])):
  try:
    reviews.iloc[row, 11] = emoji.emojize(reviews.iloc[row, 11])
  except:
    pass

reviews = reviews[['2', '3', '4', '5', '10']]
reviews.columns = ['date', 'rating', 'like', 'dislike', 'review']
reviews.dropna(axis=0, subset=['review'], inplace=True)

#전처리에 필요한 파일 다운로드
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

#소문자로 통일 및 무의미한 공백 제거
def remove_whitespace(text):
    return  " ".join(text.split())

reviews['review_new'] = reviews['review'].str.lower()
reviews['review_new'] = reviews['review_new'].apply(remove_whitespace)

#토큰화
reviews['review_new'] = reviews['review_new'].apply(lambda X: sent_tokenize(X))
for row in tqdm.tqdm(range(reviews.shape[0])):
  try:
    ele_list = []
    for ele in reviews['review_new'][row]:
      ele_list.extend(WordPunctTokenizer().tokenize(ele))
      ele_list = [word for word in ele_list if len(word) >= 3]
    reviews['review_new'][row] = ele_list
  except:
    pass

#불용어(stopwords) 처리
def remove_stopwords(text):
    result = []
    for token in text:
        if token not in stopwords.words('english'):
            result.append(token)
    return result

reviews['review_new'] = reviews['review_new'].apply(remove_stopwords)

#punctuation 제거
def remove_punct(text):
    tokenizer = RegexpTokenizer(r"\w+|\$[\d\.]+|\S+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst

reviews['review_new'] = reviews['review_new'].apply(remove_punct)

# lemmatization and stemming
def lemmatization(text):
    result = []
    wordnet = WordNetLemmatizer()
    for token, tag in pos_tag(text):
        pos = tag[0].lower()

        if pos not in ['a', 'r', 'n', 'v']:
            pos = 'n'

        result.append(wordnet.lemmatize(token, pos))

    return result

def stemming(text):
    porter = PorterStemmer()

    result = []
    for word in text:
        result.append(porter.stem(word))
    return result

reviews['review_new'] = reviews['review_new'].apply(lemmatization)
reviews['review_new'] = reviews['review_new'].apply(stemming)