from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#데이터 불러오기
train_data = pd.read_csv('/content/sample_data/watcha.csv')
train_data = train_data[['Datetime', 'Username', 'Text']]

#불용어 추가 및 처리
stopwords = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/korean_stopwords.txt").values.tolist()
store_stopwords = ['듯함', '정말', '앱내', '제발', '계속', '너무', '진짜', '좀', '그냥', '그리고', '잘', '왓챠', '더', '다시', '하지만', '이렇게', '하고', '보려고', '해서', '해도', '특히', '제대로']
for word in store_stopwords:
    stopwords.append(word)

#모델에 맞는 별도 텍스트 전처리
def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')  # 정규 표현식 처리
    result = hangul.sub('', text)
    okt = Okt()  # 형태소 추출
    return nouns

#벡터화
vect = CountVectorizer(tokenizer = lambda x: text_cleaning(x))
bow_vect = vect.fit_transform(train_data['content'].tolist()) # 각 단어의 리뷰별 등장 횟수
word_list = vect.get_feature_names() # 단어 리스트
count_list = bow_vect.toarray().sum(axis=0) # 각 단어가 전체 리뷰중에 등장한 총 횟수

tfidf_vectorizer = TfidfTransformer()
tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)

x = tf_idf_vect
y = train_data['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)


#로지스틱 회귀분석
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

lr = LogisticRegression(random_state = 0)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print('accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('precision: %.2f' % precision_score(y_test, y_pred))
print('recall: %.2f' % recall_score(y_test, y_pred))
print('F1: %.2f' % f1_score(y_test, y_pred))


# confusion matrix
from sklearn.metrics import confusion_matrix

confu = confusion_matrix(y_true = y_test, y_pred = y_pred)

plt.figure(figsize=(4, 3))
sns.heatmap(confu, annot=True, annot_kws={'size':17}, cmap='OrRd', fmt='.10g')
plt.title('Confusion Matrix')
plt.show()


#감성분석 상관계수 저장 및 시각화
plt.figure(figsize=(10, 8))
plt.bar(range(len(lr.coef_[0])), lr.coef_[0])
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse = True)[:5])
print(sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse = True)[-5:])

coef_pos_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse = True)
coef_neg_index = sorted(((value, index) for index, value in enumerate(lr.coef_[0])), reverse = False)

invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}