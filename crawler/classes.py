import os
import logging
import numpy as np
import pandas as pd
import pickle as pkl
import tweepy as tw
import threading
from pathlib import Path
# from tqdm import tqdm


# CLASSES
class TwitterAuthenticator:
    def __init__(self, credentials: dict):
        self._consumer_key = credentials.get('consumer_key')
        self._consumer_key_secret = credentials.get('consumer_key_secret')
        self._access_token = credentials.get('access_token')
        self._access_token_secret = credentials.get('access_token_secret')

        self._auth = tw.OAuthHandler(self.consumer_key, self.consumer_key_secret)
        self._auth.set_access_token(self.access_token, self.access_token_secret)

    @property
    def consumer_key(self):
        return self._consumer_key

    @consumer_key.setter
    def consumer_key(self, value: str):
        self._consumer_key = value

    @property
    def consumer_key_secret(self):
        return self._consumer_key_secret

    @consumer_key_secret.setter
    def consumer_key_secret(self, value: str):
        self._consumer_key_secret = value

    @property
    def access_token(self):
        return self._access_token

    @access_token.setter
    def access_token(self, value: str):
        self._access_token = value

    @property
    def access_token_secret(self):
        return self._access_token_secret

    @access_token_secret.setter
    def access_token_secret(self, value: str):
        self._access_token_secret = value

    @property
    def auth(self):
        return self._auth


class TwitterCrawler:
    def __init__(self, credentials: dict):
        self._twitter_authenticator = TwitterAuthenticator(credentials=credentials)
        self._api = tw.API(
            self._twitter_authenticator.auth,
            wait_on_rate_limit=True,
            wait_on_rate_limit_notify=True
        )

    @property
    def api(self):
        return self._api

    def get_user_tweets(self, user_id: int = None, num_tweets: int = 10):
        """
        user_id=None - your timeline
        num_tweets=10 - number of tweets to get
        """
        user_tweets = []
        for tweet in tw.Cursor(self.api.user_timeline, id=user_id).items(num_tweets):
            user_tweets.append(tweet)
        return user_tweets

    def get_user_friends_list(self, user_id: int):
        return np.array(self.api.friends_ids(user_id=user_id), dtype=np.int64)

    def get_user_followers_list(self, user_id: int):
        return np.array(self.api.followers_ids(user_id=user_id), dtype=np.int64)

    def check_user(self, uid: int) -> bool:
        user_valid = True
        try:
            # Check if the user exists and if its not protected
            user = self.api.get_user(uid)
            if user.protected:
                user_valid = False
        except tw.error.TweepError as tw_e:
            user_valid = False
        return user_valid

    def add_to_black_list(self, uid: int, black_list: list, lock: threading.Lock, save_file: str) -> None:
        black_list = np.append(black_list, uid)
        with lock:
            with open(save_file, 'wb') as pkl_out:
                pkl.dump(black_list, pkl_out)

    def crawl_network(self, uids: np.ndarray, network_name: str, lock: threading.Lock, logger: logging.Logger = None):
        # 1) In the begining we assume that all the UIDs are valid
        uncrawled_uids = uids
        uncrawled_uids.sort()

        # 2) Create the lists to be populated
        reliability_coeffs = list()
        followers = list()
        friends = list()

        # 4) Create Directories
        # > Black list directory
        blk_list_dir = Path('./black_list')
        if not blk_list_dir.is_dir():
            os.makedirs(blk_list_dir)

        # > Backup directory
        backup_dir = Path('./backup')
        if not backup_dir.is_dir():
            os.makedirs(backup_dir)

        # > Temp directory
        temp_dir = Path('./temp')
        if not temp_dir.is_dir():
            os.makedirs(temp_dir)

        # 5) Create Files
        # > Black list file
        network_blk_list_file = blk_list_dir / f'black_list_uids.pkl'

        # > Backup file
        network_temp_crawled_df_file = temp_dir / f'{network_name}_temp_crawled_df.pkl'

        # 6) Clean the already crawled UIDs
        crawled_uids = np.array([])
        if network_temp_crawled_df_file.is_file():
            try:
                # 6.1) load the temp Data Frame
                temp_df = pd.read_pickle(network_temp_crawled_df_file)

                crawled_uids = temp_df.uids.values
                reliability_coeffs  = list(temp_df.r.values)
                followers = list(temp_df.followers.values)
                friends = list(temp_df.friends.values)
            except Exception as err:
                if isinstance(logger, logging.Logger):
                    with lock:
                        logger.exception(f'ERROR (thread id: {threading.get_ident()}): {err}')
                        crawled_uids = np.array([])
            else:
                # 3.5.2) Subract the temporal crawled UIDs from the current UIDs
                uncrawled_uids = np.setdiff1d(uncrawled_uids, crawled_uids)

        # 3.6) Clean the unvalid UIDs from the black list
        black_list_uids = np.array([])
        if network_blk_list_file.is_file():
            try:
            # 3.6.1) Load the UIDs from the black list
                with lock:
                    with network_blk_list_file.open(mode='rb') as pkl_in:
                        black_list_uids = pkl.load(pkl_in)
            except Exception as err:
                if isinstance(logger, logging.Logger):
                    with lock:
                        logger.exception(f'ERROR (thread id: {threading.get_ident()}): {err}')
            else:
                # 3.6.2) Subract UIDs in the black list from the current UIDs
                uncrawled_uids = np.setdiff1d(uncrawled_uids, np.array(black_list_uids))

        # 3.7) If there are any UIDs left
        if uncrawled_uids.size:
            n_crawled = crawled_uids.size
            for idx, uid in enumerate(uncrawled_uids):
                progress = np.round(100 * (idx + n_crawled) / uncrawled_uids.size, 1)
                # CALCULATE ETA
                time_outs_left = np.round(int((len(uncrawled_uids) - idx) / 15)) - 1
                if time_outs_left < 0:
                    time_outs_left = 0

                if isinstance(logger, logging.Logger):
                    with lock:
                        logger.info(f'STATUS (thread id: {threading.get_ident()}): {idx + n_crawled}/{uids.size} ({progress}%), ETA < {time_outs_left} time outs left ({time_outs_left * 15} minutes)')

                if self.check_user(uid=uid):
                    # if we got here - the user exists and it's not protected, so we can try to get the follower/friend lists
                    followers_arr = np.array(self.api.followers_ids(user_id=uid), dtype=np.int64)
                    friends_arr = np.array(self.api.friends_ids(user_id=uid), dtype=np.int64)

                    try:
                        n_followers = followers_arr.size
                        n_friends = friends_arr.size

                        reliability_coeff = n_followers / (n_followers + n_friends)
                    except ZeroDivisionError as err:
                        if isinstance(logger, logging.Logger):
                            with lock:
                                logger.info(f'ERROR (thread id: {threading.get_ident()}): {err}')
                        self.add_to_black_list(uid=uid, black_list=black_list_uids, lock=lock, save_file=network_blk_list_file)
                    else:
                        if isinstance(logger, logging.Logger):
                            with lock:
                                logger.info(f'STATUS (thread id: {threading.get_ident()}): user {uid} -> r = {n_followers} / ({n_followers} + {n_friends}) = {reliability_coeff}')

                        crawled_uids = np.append(crawled_uids, uid)
                        reliability_coeffs.append(reliability_coeff)
                        followers.append(followers_arr)
                        friends.append(friends_arr)
                        logger.info(f'uids: {crawled_uids.size}, r: {len(reliability_coeffs)}, followers: {len(followers)}, friends: {len(friends)}')
                        temp_df = pd.DataFrame({'uids': crawled_uids, 'r': reliability_coeffs, 'followers': followers, 'friends': friends})
                        temp_df.to_pickle(network_temp_crawled_df_file)
                else:
                    self.add_to_black_list(uid=uid, black_list=black_list_uids, lock=lock, save_file=network_blk_list_file)
                    continue
        else:
            if isinstance(logger, logging.Logger):
                with lock:
                    logger.info(f'STATUS (thread id: {threading.get_ident()}): No residual uids to crawl!')
        if isinstance(logger, logging.Logger):
            with lock:
                logger.info(f'WORK DONE (thread id: {threading.get_ident()}): Crawl finished!')

        return pd.DataFrame({'uids': crawled_uids, 'r': reliability_coeffs, 'followers': followers, 'friends': friends})
