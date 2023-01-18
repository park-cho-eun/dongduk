install.packages(c("tidytext","dplyr","stringr","ggplot2","wordcloud2"))
library(tidytext)
library(tidyselect)
library(tidytex)
library(tidyr)
library(tokenizers)
library(dplyr)
library(stringr)
library(ggplot2)
library(wordclould)
library(wordcloud2)
library(KoNLP)
library(tidyr) #separate_rows()함수
library(widyr) #pairwise_cor()함수
library(tidygraph) #as_tbl_graph()함수
library(ggraph) #ggraph()함수
library(ldatuning) #topic model
library(scales)
library(tibble)

df <- read.csv("/Users/leeseeeun/Downloads/Contest2022/CSVfile/app_onlyreview/twits.csv")

#########################긍정 리뷰 분석 ################################
word_noun <- df %>% filter(평점>3) %>%
  unnest_tokens(input = 리뷰,
                output = 단어,
                token = extractNoun,
                drop = F)

#텍스트 변환(시각화 결과에 따라 조정 필요)
df$리뷰 <- gsub("멘","맨",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("넷플","넷플릭스",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("파티","왓챠파티",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("#더블트러블","더블트러블",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("첨","처음",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("#"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("_"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("‘","",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("‘시맨틱시멘틱에러","시맨틱에러",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("'시맨틱시멘틱에러","시맨틱에러",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("’","",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("11시"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("새빛"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("불편"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("4월"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("흡툭죽"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("분철"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("그레이태그"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("분철해드"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("전쟁"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("링크"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("<","",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("고싶은거"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("〈","",df$리뷰) #오탈자 정정
df$리뷰 <- gsub(">","",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("전쟁"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("분할"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("https"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("모집합니다"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("잇음","있음",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("공유"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("불편"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("더블","더블트러플",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("추다"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("마이"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("10분"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("3500원"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("초반"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("어쩌구"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("끊기다"," ",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("기능","성능",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("넷플릭스넷플릭스릭스","넷플릭스",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("넷플릭스릭스","넷플릭스",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("아이치","아이치이",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("넷플","넷플릭스",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("디플","디즈니플러스",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("딪플","디즈니플러스",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("쿠플","쿠팡플레이",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("튜브프리미엄","유튜브",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("시즌4","시즌",df$리뷰) #중복의미 단어 병합
df$리뷰 <- gsub("왓챠|왓챠피디아","",df$리뷰) #제품군이나 제품명을 나타내는 단어 제거
df$리뷰 <- gsub("넷플릭스릭스","넷플릭스",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("시맨틱 에러","시맨틱에러",df$리뷰) #오탈자 정정
df$리뷰 <- gsub("시맨틱","시맨틱에러",df$리뷰) #오탈자 정정

#사용자 사전에 고유명사 추가
buildDictionary(ext_dic = 'woorimalsam',user_dic = data.frame('OTT', 'ncn'),replace_usr_dic = T)

#리뷰 id 생성
df <- unique(df) #중복 리뷰 제거
df$id <- rownames(df) #행번호로 id생성

# 리뷰 데이터를 품사 태그로 토큰화
df1 <- df %>% unnest_tokens(input = 리뷰,
                            output = 품사태그,
                            token = SimplePos09,           
                            drop = F)

# 리뷰에서 명사/형용사/동사 단어 추출
df2 <- df1 %>%
  separate_rows(품사태그, sep= "[+]") %>%            # 품사태그 단위로 분리
  filter(str_detect(품사태그, "/n|/p")) %>%          # 명사/동사/형용사추출
  mutate(단어 = ifelse(str_detect(품사태그, "p"),     # 동사와 형용사 추출
                     str_replace(품사태그, "/.*$", "다"),  # "~다" 추가
                     str_remove(품사태그, "/.*$"))) %>%    # 태그 제거
  filter(str_count(단어) >= 2)                           # 2글자 이상 추출

#파이계수 구하기
word_cors <- df2 %>%
  add_count(단어) %>%
  filter(n >= 20) %>%
  pairwise_cor(item = 단어, feature = id, sort = T)

# 제품 속성 또는 소비자 혜택 관련 관심단어 지정
특징 <- c("디즈니플러스", "웨이브", "쿠팡플레이", "티빙",
        "드라마코리아", "애플티비", "라프텔", "애니",
        "드라마", "영화", "취향", "컨텐츠", "종류", "정체성", "다양")

# 관심단어와 연관성 높은 상위 10개 단어 추출
major_cors <- word_cors %>%
  filter(item1 %in% 특징) %>%
  group_by(item1) %>%
  slice_max(correlation, n = 5)

#파이계수 기반 단어 네트워크 시각화 (주요 단어 major_cors)
#데이터 준비
graph_cors <- major_cors  %>%
  filter(correlation >= 0.00001) %>%                #파이계수 필터링 기준(조정필요)
  as_tbl_graph(directed = F) %>%
  mutate(centrality = centrality_degree(),       # 중심성
         group = as.factor(group_infomap()))     # 커뮤니티
#그래프 그리기
ggraph(graph_cors, layout = "fr") +              # 레이아웃
  geom_edge_link(color = "gray50",               # 엣지 색깔
                 aes(edge_alpha = correlation,   # 엣지 명암
                     edge_width = correlation),  # 엣지 두께
                 show.legend = F) +              # 범례 삭제
  scale_edge_width(range = c(1, 4)) +            # 엣지 두께 범위
  geom_node_point(aes(size = centrality,         # 노드 크기
                      color = group),            # 노드 색깔
                  show.legend = F) +             # 범례 삭제
  scale_size(range = c(2, 5)) +                 # 노드 크기 범위
  geom_node_text(aes(label = name),              # 텍스트 표시
                 repel = T,                      # 노드밖 표시
                 size = 3,                       # 텍스트 크기
                 family = "AppleGothic") +       # 폰트
  theme_graph()                                  # 배경 삭제

# 특정 단어를 포함하는 리뷰 원문 확인하기
df %>% filter(str_detect(리뷰, "조인")) %>% head(20) %>% pull(리뷰,평점)


### Part 3. 토픽모형을 적용하여 제품리뷰의 주요 주제 탐색 #####################

# 명사 추출
noun_df <- df %>%
  filter(str_count(리뷰, boundary("word")) >= 10) %>%  # 짧은 리뷰 제거
  unnest_tokens(input = 리뷰,                         # 명사 추출
                output = word,
                token = extractNoun,
                drop = F) %>% filter(str_count(word) > 1) #한 단어 이상


# 리뷰 내 중복 단어 제거
unique_noun_df <- noun_df %>%
  group_by(id) %>%                                     # 중복 단어 제거
  distinct(word, .keep_all = T) %>%
  ungroup() %>%
  select(id, word)

# 문서별 단어 빈도 구하기
count_word <- unique_noun_df  %>%
  count(id, word, sort = T)

# DTM 만들기
dtm_df <- count_word %>%
  cast_dtm(document = id, term = word, value = n)

# LDA 토픽모형 적용
models_df <- FindTopicsNumber(dtm = dtm_df,
                              topics = 2:20,
                              return_models = T,
                              control = list(seed = 1234))

# 성능 지표 그래프
FindTopicsNumber_plot(models_df)

# 토픽모형 수에 맞게 추출
lda_model <- models_df %>% 
  filter(topics == 7) %>%    # 토픽 수 지정
  pull(LDA_model) %>%         # 모델 추출
  .[[1]]                      # list 추출

# 토픽별 단어출현 확률(beta) 추출
term_topic <- tidy(lda_model, matrix = "beta")

# 토픽별 beta 상위 단어 추출
term_topic %>%
  group_by(topic) %>%
  slice_max(beta, n = 15)

# 제품 속성 또는 소비자 혜택 관련 관심단어 지정
특징 <- c("디즈니플러스", "웨이브", "쿠팡플레이", "티빙",
        "드라마코리아", "애플티비", "라프텔", "애니",
        "드라마", "영화", "취향", "컨텐츠", "종류", "정체성", "다양")

# 불용어 목록 생성
stopword_lda <- c("alarm","clock","언제","다음","생각","^ㅋ","진짜","이거","사랑","친구",
                  "우리","co","watcha","face","heart","with","smiling","해서","궁금","엄마",
                  "button","play","오늘","시작","22.","10","다들","그거","ㅠㅠ","사람",
                  "^ㅋ^ㅋ^ㅋ^ㅋ^ㅋ","진심","기억","그것","질문","done","hourglass","알리",
                  "20","재밌게","모두","마지","30","11","31","12","02.","01.","시간")

# 불용어 제외 후 상위 단어 추출
top_term_topic <- term_topic %>%
  filter(!term %in% stopword_lda) %>%
  group_by(topic) %>%
  arrange(-beta,.by_group = TRUE) %>%
  slice(1:6)

# 토픽별 주요 단어 막대그래프 생성
ggplot(top_term_topic, aes(x = reorder_within(term, beta, topic),
                           y = beta,
                           fill = factor(topic))) +
  geom_col(show.legend = F) +
  facet_wrap(~ topic, scales = "free", ncol = 3) +
  coord_flip() +
  scale_x_reordered() +
  labs(x = NULL) +
  theme(text = element_text(family = "AppleGothic"))


# 특정 단어를 포함하는 리뷰 원문 확인하기
df %>% filter(str_detect(리뷰, "정체")) %>% head(100) %>% pull(리뷰,평점)




######################################################################

# 주요 단어 추출하기
word_noun <- df %>% filter(평점==0) %>%
  unnest_tokens(input = 리뷰,
                output = 단어,
                token = extractNoun,
                drop = F)

# 단어 빈도 구하기
frequency <- word_noun %>%
  count(단어, sort = T) %>% # 단어 빈도 구해 내림차순 정렬
  filter(str_count(단어) > 1) # 두 글자 이상만 남기기

# 상위 단어 추출
frequency %>% head(30)

# 불용어 목록 생성
stopword_noun <- c("alarm","clock","언제","다음","생각","^ㅋ","진짜","이거","사랑","친구",
                   "우리","co","watcha","face","heart","with","smiling","해서","궁금","엄마",
                   "button","play","오늘","시작","22.","10","다들","그거","ㅠㅠ","사람",
                   "^ㅋ^ㅋ^ㅋ^ㅋ^ㅋ","진심","기억","그것","질문","done","hourglass","알리",
                   "20","재밌게","모두","마지","30","11","31","12","02.","01.","시간")

# 주요 단어 목록 만들기
top30 <- frequency %>%
  filter(!단어 %in% stopword_noun) %>%
  head(30)
top30

# 막대 그래프 생성 - 기본 버전
ggplot(top30, aes(x = reorder(단어, n), y = n)) +
  geom_col() +
  coord_flip()

# 막대 그래프 생성 - 수정 버전
ggplot(top30, aes(x = reorder(단어, n), y = n)) +
  geom_col(fill='aquamarine3') +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1700)) +
  labs(title = "트위터",
       subtitle = "긍정 트윗 언급 빈도 Top 30",
       x = NULL) +
  theme_minimal() +
  theme(text = element_text(family = "AppleGothic", size = 12),
        plot.title = element_text(size = 14, face = "bold"),   # 제목 폰트
        plot.subtitle = element_text(size = 13))         # 부제목 폰트

# 워드크라우드 생성 - 기본 버전
top30 %>% wordcloud2()

# 워드크라우드 생성 - 수정 버전
top30 %>% wordcloud2(size = 1.2, color = "random-light",
                     fontFamily = 'AppleGothic', minRotation = -pi/2,
                     maxRotation = -pi/2)

# 특정 단어를 포함하는 리뷰 원문 확인하기
df %>% filter(str_detect(X, "영화")) %>% head(10) %>% pull(리뷰,평점)









# 토픽 모형에 이름 달기(선택적) : 원래는 번호만 뜨는데 -> 이름 나오게 하기
######################################################################
# 토픽 이름 목록 만들기
name_topic <- tibble(topic = 1:7,
                     name = c("콘텐츠",
                              "독점 콘텐츠",
                              "콘텐츠 업데이트 알림",
                              "시아준수",
                              "계정 쉐어",
                              "시맨틱 에러",
                              "경쟁사"))

# 토픽 이름 결합하기
top_term_topic_name <- top_term_topic %>%
  left_join(name_topic, name_topic, by = "topic")

top_term_topic_name

######################################################################

# 토픽별 주요 단어 막대그래프 생성
ggplot(top_term_topic_name,
       aes(x = reorder_within(term, beta, topic),
           y = beta,
           fill = factor(topic))) +
  geom_col(show.legend = F) +
  facet_wrap(~ name, scales = "free", ncol = 3) +
  coord_flip() +
  scale_x_reordered() +
  labs(x = NULL, y = NULL) +
  theme_minimal() +
  theme(text = element_text(family = "AppleGothic"),
        plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12))  # x축 눈금 삭제
