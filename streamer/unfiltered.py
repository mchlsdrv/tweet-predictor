import os
from os import (
    getcwd as current_dir,
    makedirs as create_dirs
)
from os.path import (
    join as make_path,
    exists as path_exists
)

import pandas as pd
from pandas import (
    Series as S,
    DataFrame as D,
    read_csv as csv_2_df,
    read_json as json_2_df
)

import numpy as np
from numpy import (
    array as np_arr,
    linspace as np_make_range,
    newaxis as np_nx,
)

import pickle as pkl
from pickle import (
    load as pkl_load,
    dump as pkl_dump
)

import tweepy as tw
from tweepy import (
    OAuthHandler as tw_authenticate,
    API as tw_api,
    Stream as tw_Stream
)

import textblob as tb
from textblob.blob import (
    TextBlob as TB
)

# conda install -c coecms dataset
import dataset as ds
from dataset import (
    connect as ds_connect,
)

# pip install datafreeze
# pip install normality
import datafreeze as dfz
from datafreeze import (
    freeze as db_freeze
)


class StreamListener(tw.StreamListener):
    def __init__(self, include_replies, db_table):
        super().__init__()
        self.include_replies = include_replies
        self.db_table = db_table

    def on_status(self, status):
        if self.include_replies:
            self.analyze_status(status)
        elif not status.in_reply_to_screen_name:
            self.analyze_status(status)

    def on_error(self, status_code):
        if status_code == 420:
            return False

    def analyze_status(self, status):
        # Tweet info
        status_id_str = status.id_str

        status_in_reply_to_status_id_str = status.in_reply_to_status_id_str
        if status_in_reply_to_status_id_str is None:
            status_in_reply_to_status_id_str = 'None'
        status_text = status.text

        status_text_truncated = status.truncated

        status_created_at = status.created_at

        ## - counters
        status_retweets_cnt = status.retweet_count

        status_fav_cnt = status.favorite_count

        status_reply_cnt = status.reply_count

        status_quote_cnt = status.quote_count

        # Source User info
        ## - user personal info
        src_user_id_str = status.user.id_str

        src_user_name = status.user.screen_name

        src_user_created_at = status.user.created_at

        src_user_url = status.user.url

        src_user_description = status.user.description

        src_user_verified = status.user.verified

        src_user_location = status.user.location

        src_user_lang = status.user.lang

        ## - user counters info
        src_user_follower_cnt = status.user.followers_count

        src_user_friend_cnt = status.user.friends_count

        src_user_listed_cnt = status.user.listed_count

        src_user_favourites_count = status.user.favourites_count

        src_user_statuses_count = status.user.statuses_count

        ## - user profile info
        src_user_prof_bg_color = status.user.profile_background_color

        src_user_prof_bg_title = status.user.profile_background_tile

        src_user_prof_link_color = status.user.profile_link_color

        src_user_prof_text_color = status.user.profile_text_color

        src_user_prof_default_img = status.user.default_profile_image

        src_user_prof_img_url = status.user.profile_background_image_url

        dst_user_id_str = status.in_reply_to_user_id_str
        dst_user_name = status.in_reply_to_screen_name

        # Analyze the tweet
        sentiment = TB(status_text)

        status_polarity = sentiment.polarity

        status_subjectivity = sentiment.subjectivity
        print(status_text)
        try:
            self.db_table.insert(
                dict(
                    tweet_created_at=status_created_at,

                    tweet_id=status_id_str,
                    to_tweet_id=status_in_reply_to_status_id_str,

                    from_user_id=src_user_id_str,
                    from_user_name=src_user_name,

                    to_user_id=dst_user_id_str,
                    to_user_name=dst_user_name,

                    src_user_verified=src_user_verified,
                    src_user_follower_cnt=src_user_follower_cnt,
                    src_user_friend_cnt=src_user_friend_cnt,
                    src_user_listed_cnt=src_user_listed_cnt,
                    src_user_statuses_count=src_user_statuses_count,
                    src_user_favourites_count=src_user_favourites_count,

                    src_user_created_at=src_user_created_at,
                    src_user_url=src_user_url,
                    src_user_description=src_user_description,
                    src_user_location=src_user_location,
                    src_user_language=src_user_lang,

                    src_user_prof_bg_color=src_user_prof_bg_color,
                    src_user_prof_bg_title=src_user_prof_bg_title,
                    src_user_prof_link_color=src_user_prof_link_color,
                    src_user_prof_text_color=src_user_prof_text_color,
                    src_user_prof_default_img=src_user_prof_default_img,
                    src_user_prof_img_url=src_user_prof_img_url,

                    tweet_content=status_text,
                    tweet_content_truncated=status_text_truncated,

                    retweet_cnt=status_retweets_cnt,
                    fav_cnt=status_fav_cnt,
                    reply_cnt=status_reply_cnt,
                    quote_cnt=status_quote_cnt,

                    status_polarity=status_polarity,
                    status_subjectivity=status_subjectivity
                )
            )
        except Exception as e:
            print(e)


class StreamListener(tw.StreamListener):
    def __init__(self, include_replies, db_table):
        super().__init__()
        self.include_replies = include_replies
        self.db_table = db_table

    def on_status(self, status):
        if self.include_replies:
            self.analyze_status(status)
        elif not status.in_reply_to_screen_name:
            self.analyze_status(status)

    def on_error(self, status_code):
        if status_code == 420:
            return False

    def analyze_status(self, status):
        # Tweet info
        status_id_str = status.id_str

        status_in_reply_to_status_id_str = status.in_reply_to_status_id_str
        if status_in_reply_to_status_id_str is None:
            status_in_reply_to_status_id_str = 'None'
        status_text = status.text

        status_text_truncated = status.truncated

        status_created_at = status.created_at

        ## - counters
        status_retweets_cnt = status.retweet_count

        status_fav_cnt = status.favorite_count

        status_reply_cnt = status.reply_count

        status_quote_cnt = status.quote_count

        # Source User info
        ## - user personal info
        src_user_id_str = status.user.id_str

        src_user_name = status.user.screen_name

        src_user_created_at = status.user.created_at

        src_user_url = status.user.url

        src_user_description = status.user.description

        src_user_verified = status.user.verified

        src_user_location = status.user.location

        src_user_lang = status.user.lang

        ## - user counters info
        src_user_follower_cnt = status.user.followers_count

        src_user_friend_cnt = status.user.friends_count

        src_user_listed_cnt = status.user.listed_count

        src_user_favourites_count = status.user.favourites_count

        src_user_statuses_count = status.user.statuses_count

        ## - user profile info
        src_user_prof_bg_color = status.user.profile_background_color

        src_user_prof_bg_title = status.user.profile_background_tile

        src_user_prof_link_color = status.user.profile_link_color

        src_user_prof_text_color = status.user.profile_text_color

        src_user_prof_default_img = status.user.default_profile_image

        src_user_prof_img_url = status.user.profile_background_image_url

        dst_user_id_str = status.in_reply_to_user_id_str
        dst_user_name = status.in_reply_to_screen_name

        # Analyze the tweet
        sentiment = TB(status_text)

        status_polarity = sentiment.polarity

        status_subjectivity = sentiment.subjectivity
        print(status_text)
        try:
            self.db_table.insert(
                dict(
                    tweet_created_at=status_created_at,

                    tweet_id=status_id_str,
                    to_tweet_id=status_in_reply_to_status_id_str,

                    from_user_id=src_user_id_str,
                    from_user_name=src_user_name,

                    to_user_id=dst_user_id_str,
                    to_user_name=dst_user_name,

                    src_user_verified=src_user_verified,
                    src_user_follower_cnt=src_user_follower_cnt,
                    src_user_friend_cnt=src_user_friend_cnt,
                    src_user_listed_cnt=src_user_listed_cnt,
                    src_user_statuses_count=src_user_statuses_count,
                    src_user_favourites_count=src_user_favourites_count,

                    src_user_created_at=src_user_created_at,
                    src_user_url=src_user_url,
                    src_user_description=src_user_description,
                    src_user_location=src_user_location,
                    src_user_language=src_user_lang,

                    src_user_prof_bg_color=src_user_prof_bg_color,
                    src_user_prof_bg_title=src_user_prof_bg_title,
                    src_user_prof_link_color=src_user_prof_link_color,
                    src_user_prof_text_color=src_user_prof_text_color,
                    src_user_prof_default_img=src_user_prof_default_img,
                    src_user_prof_img_url=src_user_prof_img_url,

                    tweet_content=status_text,
                    tweet_content_truncated=status_text_truncated,

                    retweet_cnt=status_retweets_cnt,
                    fav_cnt=status_fav_cnt,
                    reply_cnt=status_reply_cnt,
                    quote_cnt=status_quote_cnt,

                    status_polarity=status_polarity,
                    status_subjectivity=status_subjectivity
                )
            )
        except Exception as e:
            print(e)


if __name__ == '__main__':
    CURRENT_DIR_PATH = current_dir()
    twitter_info_file_path = make_path(CURRENT_DIR_PATH, 'twitter_info.txt')
    # if path_exists(twitter_info_file_path):
        # with open(twitter_info_file_path, 'r') as tw_f:
#             data_list = tw_f.read().split(', ')
    CONSUMER_KEY = 'HONJd5HL8yxE4W882u0ksrrmx'
    CONSUMER_SECRET = 'afjYCOxGwSLFqSn0mRddDQFQkVhBxL21xOfPg2DuqECm2Y2PFS'
    ACCESS_TOKEN = '1413781861390864386-4eIvgbXcm2Rre8hpNb6bh5Gm8waoNe'
    ACCESS_TOKEN_SECRET = 'jJKWHXXGWYM7wQDQSDBQVAgaDuo5j9T603f7HbEW7jKsG'
    print(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    auth = tw_authenticate(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tw_api(auth)

    output_data_dir = make_path(CURRENT_DIR_PATH, 'OutputData')
    if not path_exists(output_data_dir):
        create_dirs(output_data_dir)

    DB_NAME = 'unfiltered_tweets'
    db_file_path = "sqlite:///{}".format(make_path(output_data_dir, DB_NAME + '.db'))
    db = ds_connect(db_file_path)

    TWEETS_TABLE = 'tweets_table'
    db_tweets_table = db[TWEETS_TABLE]

    sl = StreamListener(include_replies=True, db_table=db_tweets_table)
    stream = tw_Stream(
        auth=api.auth,
        listener=sl
    )
    TWEET_FILTER_LIST = []
    while True:
        try:
            # stream.filter(track=TWEET_FILTER_LIST, stall_warnings=True)
            stream.sample()
        except Exception as e:
            print(e)