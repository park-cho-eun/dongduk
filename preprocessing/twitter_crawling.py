#snscrapper_https://github.com/JustAnotherArchivist/snscrape.git

import pandas as pd
import snscrape.modules.twitter as sntwitter
import tqdm

#2022년 기준 '왓챠'키워드 들어간 트윗 모두 수집
for num in range(2, 13):
    if num < 10:
        num = '0' + str(num)
    if num == '02':
        date = '28'
    elif num in ['04', '06', '09', '10']:
        date = '30'
    else:
        date = '31'

    twt_list = []
    for tweet in tqdm.tqdm(sntwitter.TwitterSearchScraper(
            '왓챠 since:2022-' + str(num) + '-01 until:2022-' + str(num) + '-' + str(date)).get_items()):
        if tweet.content.startswith('@'):
            pass
        else:
            twt_list.append([tweet.date, tweet.user.username, tweet.content])

    tweets_df1 = pd.DataFrame(twt_list, columns=['Datetime', 'Username', 'Text'])
    tweets_df1.to_csv('/content/drive/MyDrive/dd/twt/' + str(num) + '.csv')
    print('number ' + str(num) + ' done!')
