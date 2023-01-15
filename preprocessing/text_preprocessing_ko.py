'''
패키지 깃허브 주소
https://github.com/haven-jeon/PyKoSpacing.git
https://github.com/ssut/py-hanspell.git
'''

from hanspell import spell_checker
from pykospacing import Spacing
from konlpy.tag import *
import re
import tqdm
import pandas as pd

okt = konlpy.tag.Okt() #형태소 분석기
spacing = Spacing() #띄어쓰기 분석기
hangul = re.compile('[^ ㄱ-ㅣ 가-힣]') #한글위주 텍스트 선별

twit = pd.read_csv('/content/drive/MyDrive/dd/twits.csv')

texts = []
for num in tqdm.tqdm(texts.shape[0]):
  try:
    sent = twit['Text'][num]
    sent = hangul.sub('', sent)
    sent = sent.replace(" ", '')
    sent = spacing(sent)

    sent = spell_checker.check(sent) #한글 맞춤법 검사
    sent = sent.checked

    morph = okt.morphs(sent)
    sent = ' '.join(morph)
    texts.append(sent)

  except:
    texts.append(' ')

texts = pd.DataFrame(texts)
texts.to_csv('/content/drive/MyDrive/dd/twit_text.csv')