# Telegram Data Contest
Docker based - solution for Telegram [Data Cluster contest](https://contest.com/docs/data_clustering)
* git hub thirdparty folder is not full version

#### build
sudo docker build --network=host -t news_handler .

#### run
sudo docker run -p <OUTER_PORT>:<INNER_PORT> -it news_handler

