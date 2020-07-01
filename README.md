# Telegram Data Contest
Docker based - solution for Telegram [Data Cluster contest](https://contest.com/docs/data_clustering)


You can train your own models by using datasets like that https://www.kaggle.com/yutkin/corpus-of-russian-news-articles-from-lenta
( github thirdparty folder is not full version )

#### build
sudo docker build --network=host -t news_handler .

#### run
sudo docker run -p <OUTER_PORT>:<INNER_PORT> -it news_handler

