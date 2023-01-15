from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd
import tqdm

import seaborn as sns
from matplotlib import pyplot as plt


#데이터 불러오기
twit = pd.read_csv('/content/drive/MyDrive/dd/twits.csv')

#긍정/부정 데이터 분리
label_0 = twit[twit['label']==0]['Text_p']
label_0 = label_0.reset_index(drop=True)

label_1 = twit[twit['label']==1]['Text_p']
label_1 = label_1.reset_index(drop=True)

#전처리된 텍스트 Tf-idf 벡터화
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(twit[twit['label']==0]['Text_p'])

# Elbow활용해 적절한 군집수 찾기
# Inertia(군집 내 거리제곱합의 합) value (적정 군집수)
ks = range(1,15)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(4, 4))
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#11개의 군집으로 kmeans clustering 수행
true_k = 11
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
clusters = model.fit(X)

#각 군집의 벡터 평균 구하기
means = []
for num in tqdm.tqdm(range(true_k)):
  arr = [model.transform(X[x]) for x, y in zip(range(label_0.shape[0]), model.labels_) if  y == num]
  arr = np.array(arr)
  mean = arr.mean(axis=0)
  means.append(mean)

#각 군집 평균 벡터에 가장 가까운 대표 분장 추출
for k, cent in enumerate(means):
  similar = {}
  for row in tqdm.tqdm(range(label_0.shape[0])):
    vec_1 = model.transform(X[row])
    cos = cosine_similarity(vec_1, cent)
    similar[row] = cos
  Max = max(similar, key=similar.get)
  print(str(k)+'번째 군집의 대표 리뷰: '+str(label_0[Max]))

#2차원 군집 시각화
from sklearn.decomposition import TruncatedSVD
clf = TruncatedSVD(2)
Xpca = clf.fit_transform(X)

pca_df = pd.DataFrame(Xpca)
pca_df['cluster'] = clusters.labels_

axs = plt.subplots()
axs = sns.scatterplot(0, 1, hue='cluster', data=pca_df)

#3차원 군집 시각화
clf = TruncatedSVD(3)
Xpca = clf.fit_transform(X)

pca_df = pd.DataFrame(Xpca)
pca_df['cluster'] = clusters.labels_

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
X = pca_df

ax.scatter(  X.iloc[:,0]
           , X.iloc[:,1]
           , X.iloc[:,2]
           , c = X.cluster
           , s = 10
           , cmap = "rainbow"
           , alpha = 1
          )
