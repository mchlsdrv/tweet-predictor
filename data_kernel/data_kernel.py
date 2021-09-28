import sys
import os
from os import (
	makedirs
)
import yaml
import copy
import time
import re
import pathlib
from pathlib import (
	Path
)
from datetime import (
	datetime
)
import gc
import pandas as pd
import numpy as np
import logging.config
from functools import partial
import itertools
from itertools import (
	chain
)
import networkx as nx
import matplotlib.pyplot as plt


# Global Functions
def configure_output_dir(dir_path: str, default_dir_path: str, logger: logging.Logger = None) -> str:
	# Ensure that the log dir exists
	valid_dir_path = default_dir_path
	try:
		if dir_path is not None:
			log_dir = Path(dir_path)
			if not log_dir.is_dir():
				os.makedirs(log_dir)
			valid_dir_path = dir_path
	except Exception as err:
		if logger is not None:
			logger.exception(err)
	return valid_dir_path


def get_dir_path_file_name_file_type(path: pathlib.Path, logger: logging.Logger = None) -> (pathlib.Path, str, str):
	dir_name = None
	file_name_type = None
	file_name = None
	file_type = None

	if logger is not None:
		logger.info('''
	----------------------------------------
	=== get_dir_path_file_name_file_type ===
	----------------------------------------
	''')
		logger.info(f'Trying to parse the \'{path}\' path ...')

	path_str = str(path)
	try:
		# Try the windows file notation
		dir_name = Path(path_str[::-1][path_str[::-1].index('\\')+1:][::-1])
		file_name_type = path_str[::-1][:path_str[::-1].index('\\')][::-1]
	except ValueError as err:
		if logger is not None:
			logger.exception(err)
			logger.info(f'Trying to parse the \'{path}\' path in a Linux style ...')
		# If the file is not represented in windows style - try parsing it in linux notation
		try:
			dir_name = Path(path_str[::-1][path_str[::-1].index('/')+1:][::-1])
			file_name_type = path_str[::-1][:path_str[::-1].index('/')][::-1]
		except ValueError as err:
			if logger is not None:
				logger.exception(err)
				logger.info(f'Could not parse the \'{path}\' path !')
	if logger is not None:
		logger.info(f'Successfully parsed the \'{path}\' path into {file_name_type} !')

	if file_name_type is not None:
		file_name, file_type = file_name_type[:file_name_type.index('.')], file_name_type[file_name_type.index('.')+1:]

	if logger is not None:
		logger.info(f'''
The path \'{path}\' was successfully parsed into:
	- Directory name: {dir_name}
	- File name: {file_name}
	- File type: {file_type}
	''')
	return dir_name, file_name, file_type


def get_ts():
	def _clean_ts(ts):
		ts = ts[::-1]
		ts = ts[ts.index('.')+1:]
		ts = ts[::-1]
		ts = ts.replace(':', '_')
		return ts
	ts = str(datetime.now())
	return _clean_ts(ts)


def check_dir(dir_path, logger=None):
	dir_ok = False
	if isinstance(dir_path, Path) or isinstance(dir_path, WindowsPath):
		if not dir_path.is_dir():
			makedirs(dir_path)
		dir_ok = True
		if logger is not None:
			logger.info(f'The \'{dir_path}\' is valid.')
	else:
		if logger is not None:
			logger.info('ERROR in check_dir: The path to save_dir ({dir_path}) is not of type \'Path\' but of type {type(dir_path)}!')
	return dir_ok


def get_run_time(seconds):
	hours = int(seconds // 3600)
	minutes = int((seconds - hours * 3600) // 60)
	residual_seconds = int(seconds - hours * 3600 - minutes * 60)
	return f'Runtime\n\t=> {hours}:{minutes}:{residual_seconds} [H:m:s]'


def get_mem_usage(object):
	mem_in_bytes = sys.getsizeof(object)
	kilo = 1000
	mega = 1000000
	giga = 1000000000
	tera = 1000000000000
	if mem_in_bytes > tera:
		hr_mem_str = f'{mem_in_bytes // tera}.{mem_in_bytes % tera:.0f}[TB]'
	elif mem_in_bytes > giga:
		hr_mem_str = f'{mem_in_bytes // giga}.{mem_in_bytes % giga:.0f}[GB]'
	elif mem_in_bytes > mega:
		hr_mem_str = f'{mem_in_bytes // mega}.{mem_in_bytes % mega:.0f}[MB]'
	elif mem_in_bytes > kilo:
		hr_mem_str = f'{mem_in_bytes // kilo}.{mem_in_bytes % kilo:.0f}[KB]'
	else:
		hr_mem_str = f'{mem_in_bytes}[Bytes]'
	return hr_mem_str


def find_rt_uname(tw_text):
	rt_uname = re.findall(r'^@(\w+)\b', tw_text)
	return '' if not rt_uname else rt_uname[0]


def find_re_unames(tw_text):
	re_unames = re.findall(r'(?i)(?<=\bRT )@(\w+):', tw_text)


def find_mt_unames(tw_text):
	mt_unames = re.findall(r'(?i)(?<!\bRT )@(\w+)', tw_text)
	return mt_unames if mt_unames else None


def time_exec(func, logger=None):
	start_time = time.time()
	ret_val = func()

	end_time = time.time() - start_time

	info = f'''
	Run time: {get_run_time(seconds=end_time)}
	'''

	if logger is not None:
		logger.info(info)

	return  ret_val, end_time


def get_logger(configs_file_path: pathlib.Path) -> logging.Logger:
	logger = None

	with configs_file_path.open(mode='r') as f:
		configs = yaml.safe_load(f.read())
		# Check that the log directory is valid
		dir_path, file_name, file_type = get_dir_path_file_name_file_type(configs['handlers']['logfile']['filename'])
		valid_log_dir_path = Path(configure_output_dir(dir_path=dir_path, default_dir_path='./logs', logger=None))
		# Assign a valid path to the log file
		configs['handlers']['logfile']['filename'] = str(valid_log_dir_path / f'{file_name}.{file_type}')
		logging.config.dictConfig(configs)
		logger = logging.getLogger(__name__)

	return logger


# Configurations
# - Name Configurations
NET_NAME = 'unfiltered'

# - Time Configurations
DAYS = 0
HOURS = 12
MINUTES = 0
TIME_WINDOW = pd.Timedelta(f'{DAYS} days {HOURS} hours {MINUTES} minutes')

# - Activity Proportions Configurations
N = 10000
ACT_PROP = 3.6
POP_PROP = 7.2

# - Path Configurations
# > Logger configurations file path
LOGGER_CONFIGS_FILE_PATH = Path('./configs/logger_configs.yml')

# > root dirs
INPUT_DIR = Path(f'./input/{NET_NAME.upper()}')
OUTPUT_DIR = Path(f'./output/{NET_NAME.upper()}')

# > Input data in a prquet format
INPUT_PARQ_DIR = INPUT_DIR / f'parq'
POSTS_DF_PARQ_FILE_PATH = INPUT_PARQ_DIR / f'{NET_NAME}_tw_log.parq.gzip'

# > Input data in a pickle format
INPUT_PKL_DIR = INPUT_DIR / f'pkl'
POSTS_DF_PKL_FILE_PATH = INPUT_PKL_DIR / f'{NET_NAME}_clean_posts.pkl'
TOP_SOCIAL_NETWORK_FILE = INPUT_PKL_DIR / 'top_social_network_df.pkl'
TOP_POSTS_FILE = INPUT_PKL_DIR / 'top_posts_df.pkl'

# > Output data
EXPERIMENT_OUTPUT_DATA_DIR = OUTPUT_DIR / f'act_{POP_PROP}_pop_{POP_PROP}_{DAYS}D-{HOURS}H-{MINUTES}m'.replace('.', '_')
SOCIAL_NETWORK_DIR = EXPERIMENT_OUTPUT_DATA_DIR / 'social_network'
VECTORS_DIR = EXPERIMENT_OUTPUT_DATA_DIR / 'vectors'

COLUMN_NAMES = dict(
	tweet_created_at = 'ts',
	tweet_id = 'tw_id',
	to_tweet_id = 'rt_id',
	from_user_id = 'tw_uid',
	to_user_id = 'rt_uid',
	from_user_name = 'tw_uname',
	to_user_name = 'rt_uname',
	tweet_content = 'text',
	src_user_verified = 'user_verified',
	src_user_follower_cnt = 'n_followers',
	src_user_friend_cnt = 'n_friends',
	src_user_listed_cnt = 'n_lists',
	src_user_statuses_count = 'n_statuses',
	src_user_favourites_count = 'n_favorites',
	src_user_created_at = 'user_creation_date',
	src_user_description = 'user_description',
	status_polarity = 'polarity',
	status_subjectivity = 'subjectivity'
)

COLUMN_DTYPES = dict(
	tw_id='int64',
	rt_id='int64',
	tw_uid='int64',
	rt_uid='int64',
	n_followers='int16',
	n_friends='int16',
	n_lists='int16',
	n_statuses='int16',
	n_favorites='int16',
	polarity='float16',
	subjectivity='float16',
	text='str',
	user_description='str',
	user_verified='boolean',
)


def get_clean_posts_df(posts_df_parq_file_path: pathlib.Path, save_dir_path: pathlib.Path, logger: logging.Logger) -> pd.DataFrame:

	logger.info('''
	--------------------------
	=== get_clean_posts_df ===
	--------------------------
	''')
	logger.info('1) Read the file - original data')
	posts_df, exec_time = load_data_from_file(data_file_path=posts_df_parq_file_path, logger=logger)  #time_exec(func=partial(pd.read_parquet, path=posts_df_parq_file_path), logger=logger)

	logger.info(posts_df.head())

	logger.info(
	f'''
	Log Summary:
		Total number of posts: {posts_df.shape[0]}
	'''
	)

	info = f'Data types: \n{posts_df.dtypes}'
	logger.info(info)

	logger.info('2) Drop Unnecessary columns')
	logger.info('2.1) DataFrame Modification')
	columns_2_drop = set(posts_df.columns) ^ set(COLUMN_NAMES.keys())
	time_exec(func=partial(posts_df.drop, columns=columns_2_drop, axis=1, inplace=True), logger=logger)

	logger.info(
	f'''
	Posts:
		{posts_df.head()}
	'''
	)

	logger.info('2.2) DataFrame Modification')
	time_exec(func=partial(posts_df.rename, columns=COLUMN_NAMES, inplace=True), logger=logger)
	logger.info(
	f'''
	Posts:
		{posts_df.head()}
	'''
	)

	logger.info('2.3) Clean missing values')
	time_exec(func=partial(posts_df.fillna, -1, inplace=True), logger=logger)
	posts_df.loc[posts_df.loc[:, 'rt_id'] == 'None', 'rt_id'] = -1
	logger.info(posts_df.head())
	logger.info(
	f'''
	Log Summary:
		> Total number of posts: {posts_df.shape[0]}
	'''
	)

	logger.info('2.4) Change Data Types')
	posts_df, exec_time = time_exec(func=partial(posts_df.astype, dtype=COLUMN_DTYPES), logger=logger)
	posts_df.ts = pd.to_datetime(posts_df.ts, format='%Y-%m-%d %H:%M:%S')
	posts_df.user_creation_date = pd.to_datetime(posts_df.user_creation_date, format='%Y-%m-%d %H:%M:%S')

	logger.info(
	f'''
	Data types:
		{posts_df.dtypes}
	'''
	)

	logger.info('2.5) Add features')
	uname_uid_map = dict(zip(posts_df.tw_uname, posts_df.tw_uid))

	logger.info('2.5.1) re_uname')
	posts_df.loc[:, 're_unames'] = posts_df.loc[:, 'text'].apply(find_re_unames)
	posts_df.loc[:, 're_uids'] = posts_df.loc[:, 're_unames'].apply(lambda unames: None if unames is None else np.array([uname_uid_map.get(uname) for uname in unames if uname in uname_uid_map.keys()] , dtype=np.int64))

	logger.info('2.5.2) mt_unames')
	posts_df.loc[:, 'mt_unames'] = posts_df.loc[:, 'text'].apply(find_mt_unames)
	posts_df.loc[:, 'mt_uids'] = posts_df.loc[:, 'mt_unames'].apply(lambda mt_unames: None if mt_unames is None else np.array([uname_uid_map.get(mt_uname) for mt_uname in mt_unames if mt_uname in uname_uid_map.keys()], dtype=np.int64))

	logger.info('2.5.3) type')
	posts_df.loc[~posts_df.re_uids.isna(), 'type'] = 'Reply'
	posts_df.loc[(posts_df.re_uids.isna()) & (~posts_df.mt_uids.isna()), 'type'] = 'Mention'
	posts_df.loc[(posts_df.rt_uid!=-1) & (posts_df.re_uids.isna()) & (posts_df.mt_uids.isna()), 'type'] = 'Re-Tweet'
	posts_df.loc[(posts_df.rt_id!=-1) & (posts_df.rt_uid==-1) & (posts_df.re_uids.isna()) & (posts_df.mt_uids.isna()), 'type'] = '1-Click'
	posts_df.loc[(posts_df.rt_id==-1) & (posts_df.rt_uid==-1) & (posts_df.re_uids.isna()) & (posts_df.mt_uids.isna()), 'type'] = 'Tweet'
	posts_df = posts_df[
		[
		'type', 'ts',
		'tw_id', 'rt_id',
		'tw_uid', 'rt_uid', 're_uids', 'mt_uids',
		'tw_uname', 'rt_uname', 're_unames', 'mt_unames',
		'polarity', 'subjectivity',
		'user_verified',
		'n_followers', 'n_friends', 'n_lists', 'n_statuses', 'n_favorites',
		'user_creation_date', 'user_description', 'text',
		]
	]
	logger.info(
	f'''
	Posts:
		{posts_df.head()}
	'''
	)

	posts_df.to_pickle(path=str(save_dir_path / f'{NET_NAME}_clean_data.pkl'))

	return posts_df


def get_top_social_network_df(posts_df: pd.DataFrame, save_dir_path: pathlib.Path, logger: logging.Logger = None) -> pd.DataFrame:
	logger.info(f'''
	------------------------------
	=== get_top_social_network ===
	------------------------------
	''')

	start_time = time.time()

	logger.info('1) Separate the tweets into Replies, Mentions, Re-Tweets, 1-Click and Tweets')
	logger.info('1.1) Replies')
	RE = posts_df.loc[posts_df.type=='Reply'].reset_index(drop=True)  # RT @username
	logger.info(f'''
	Replies
		{RE.head()}
	''')

	logger.info('1.2) Mentions')
	MT = posts_df.loc[posts_df.type=='Mention'].reset_index(drop=True)  # text ... @username
	logger.info(f'''
	Mentions
		{MT.head()}
	''')

	logger.info('1.3) Re-Tweets')
	RT = posts_df.loc[posts_df.type=='Re-Tweet'].reset_index(drop=True) # @username ... text
	logger.info(f'''
	Re-Tweets
		{RT.head()}
	''')

	logger.info('1.4) 1-Click')
	OC = posts_df.loc[posts_df.type=='1-Click'].reset_index(drop=True) # 1-click re-tweets
	logger.info(f'''
	1-Click
		{OC.head()}
	''')

	logger.info('1.5) Tweets')
	start_time = time.time()
	TW = posts_df.loc[posts_df.type=='Tweet'].reset_index(drop=True) # not a 1-click re-tweets or RT @username replies
	end_time = time.time() - start_time

	logger.info(
	f'''
	Post Log Stats
		> N Posts: {posts_df.shape[0]}
			- N Replies: {RE.shape[0]}
			- N Mentions: {MT.shape[0]}
			- N Re-Tweets: {RT.shape[0]}
			- N 1-Clicks: {OC.shape[0]}
			- N Tweets: {TW.shape[0]}
		> Sum of all the sub type post logs = {RE.shape[0]+MT.shape[0]+RT.shape[0]+OC.shape[0]+TW.shape[0]}
		> ?= N Posts: {posts_df.shape[0]==RE.shape[0]+MT.shape[0]+RT.shape[0]+OC.shape[0]+TW.shape[0]}
		> {get_run_time(seconds=end_time)}
	'''
	)

	logger.info('2) Get the users involved in the interactions')
	logger.info('2.1) Active Users')
	replying_uids = RE.tw_uid.values
	mentioning_uids = MT.tw_uid.values
	retweeting_uids = RT.tw_uid.values
	one_clicking_uids = OC.tw_uid.values
	tweeting_uids = TW.tw_uid.values
	active_uids = posts_df.tw_uid.values

	unique_replying_uids = np.unique(replying_uids)
	unique_mentioning_uids = np.unique(mentioning_uids)
	unique_retweeting_uids = np.unique(retweeting_uids)
	unique_one_clicking_uids = np.unique(one_clicking_uids)
	unique_tweeting_uids = np.unique(tweeting_uids)
	unique_active_uids = np.unique(active_uids)

	logger.info(
	f'''
	Social Network Stats:
		> Unique Users Number: {unique_active_uids.shape[0]}
			- Replying Users Number: {unique_replying_uids.shape[0]}
			- Mentioning Users Number: {unique_mentioning_uids.shape[0]}
			- Re-Tweeting Users Number: {unique_retweeting_uids.shape[0]}
			- Tweeting Users Number: {unique_tweeting_uids.shape[0]}
			- 1-Clicked Users Number: {unique_one_clicking_uids.shape[0]}
		> Sum of all the active users = {unique_replying_uids.shape[0]+unique_mentioning_uids.shape[0]+unique_retweeting_uids.shape[0]+unique_one_clicking_uids.shape[0]+unique_tweeting_uids.shape[0]}
	'''
	)

	logger.info('2.2) Passive Users')
	replied_uids = np.concatenate(RE.re_uids.values) if RE.re_uids.values.any() else np.array([])
	mentioned_uids = np.concatenate(MT.mt_uids.values) if MT.mt_uids.values.any() else np.array([])
	retweeted_uids = RT.rt_uid.values
	one_clicked_uids = OC.rt_uid.values

	unique_replied_uids = np.unique(replied_uids)
	unique_mentioned_uids = np.unique(mentioned_uids)
	unique_retweeted_uids = np.unique(retweeted_uids)
	unique_one_clicked_uids = np.unique(one_clicked_uids)

	unique_soc_net_uids = np.unique(np.concatenate([unique_replied_uids, unique_mentioned_uids, unique_retweeted_uids, unique_one_clicked_uids]))

	logger.info(
	f'''
	Social Network Stats:
		> Unique Users Number: {unique_soc_net_uids.shape[0]}
			- Replied Users Number: {unique_replied_uids.shape[0]}
			- Mentioned Users Number: {unique_mentioned_uids.shape[0]}
			- Re-Tweeted Users Number: {unique_retweeted_uids.shape[0]}
			- 1-Clicked Users Number: {unique_one_clicked_uids.shape[0]}
		> Sum of all the passive users = {unique_replied_uids.shape[0]+unique_mentioned_uids.shape[0]+unique_retweeted_uids.shape[0]+unique_one_clicked_uids.shape[0]}
	'''
	)

	logger.info('3) Time Windows')
	time_windows = pd.to_timedelta(posts_df.ts.max() - posts_df.ts.min()) // pd.to_timedelta(TIME_WINDOW)
	logger.info(f'''
	Time Windows Configurations:
		> Number of time window slices of {TIME_WINDOW} : {time_windows}
	''')

	logger.info('4) Users Posts')
	logger.info('4.1) Active Users Posts')
	n_posts = int(ACT_PROP*time_windows)

	unique_active_uids, counts = np.unique(active_uids, return_counts=True)
	activity_map = np.array(sorted(np.array(list(zip(counts, unique_active_uids))), key=lambda x: x[0])[::-1], dtype=np.int64)

	top_active_uids = activity_map[np.argwhere(activity_map[:, 0]>n_posts).flatten(), 1]
	logger.info(f'''
	Number of users with more than {n_posts} posts:
		- {top_active_uids.shape[0]}
	'''
	)

	logger.info('4.2) Popular Users Posts')
	n_reactions = int(POP_PROP*time_windows)

	unique_popular_uids, counts = np.unique(np.concatenate([replied_uids, mentioned_uids, retweeted_uids, one_clicked_uids]), return_counts=True)
	popularity_map = np.array(sorted(np.array(list(zip(counts, unique_popular_uids))), key=lambda x: x[0])[::-1], dtype=np.int64)

	top_popular_uids = popularity_map[np.argwhere(popularity_map[:, 0]>n_reactions).flatten(), 1]
	logger.info(f'''
	Number of users with more than {n_reactions} reactions (replies, mentions, re-tweets, or 1-clicks):
		- {top_popular_uids.shape[0]}
	'''
	)

	logger.info('5) Filter based on Broadcastisity')
	I = np.intersect1d(top_popular_uids, top_active_uids)
	U = np.union1d(top_popular_uids, top_active_uids)
	IoU = I.shape[0] / U.shape[0]

	logger.info(f'''
	Post Log Broadcastisity:
		> |I| = {I.shape[0]}
		> |U| = {U.shape[0]}
		> IoU = |I| / |U| = {I.shape[0]} / {U.shape[0]} = {IoU:.2f}
		=> B = 1 - IoU = 1 - {IoU:.2f} = {1 - IoU:.2f}
	'''
	)

	logger.info('5.1) Leave posts which tweeting uid is present in U')
	start_time = time.time()
	top_posts = posts_df.loc[posts_df.tw_uid.isin(U)].reset_index(drop=True)
	top_posts_time_span = pd.to_timedelta(top_posts.ts.max() - top_posts.ts.min())
	end_time = time.time() - start_time
	logger.info(f'''
	Top Posts Log Stats:
		> Number of posts: {top_posts.shape[0]}
		> Time Span: {top_posts_time_span}
		> 'top_posts' size: {get_mem_usage(object=top_posts)}
		> {get_run_time(seconds=end_time)}
	''')

	logger.info('5.2) Leave tweets which tweeting uid is present in U')
	start_time = time.time()
	top_tweets = TW.loc[TW.tw_uid.isin(U)].reset_index(drop=True)

	top_tweets_time_span = pd.to_timedelta(top_tweets.ts.max() - top_tweets.ts.min())
	end_time = time.time() - start_time
	logger.info(f'''
	Top Tweet Log Stats:
		> Number of posts: {top_tweets.shape[0]}
		> Time Span: {top_tweets_time_span}
		> 'top_tweets' size: {get_mem_usage(object=top_tweets)}
		> {get_run_time(seconds=end_time)}
	''')

	logger.info('5.3) Leave tweets which tweeting uid and uid being re-tweeted are present in U')
	start_time = time.time()
	top_retweets = RT.loc[RT.tw_uid.isin(U) & RT.rt_uid.isin(U)].reset_index(drop=True)

	top_retweets_time_span = pd.to_timedelta(top_retweets.ts.max() - top_retweets.ts.min())
	end_time = time.time() - start_time
	logger.info(f'''
	Top 1-click re-tweet log stats
		> Number of posts: {top_retweets.shape[0]}
		> Time Span: {top_retweets_time_span}
		> 'top_retweets' size: {get_mem_usage(object=top_retweets)}
		> {get_run_time(seconds=end_time)}
	''')

	start_time = time.time()

	logger.info('5.4) Leave replies which tweeting uid is present in U')
	top_replies = RE.loc[RE.tw_uid.isin(U)].reset_index(drop=True)

	logger.info('5.5) Leave only the replied uids that are present in U')
	top_replies.loc[:, 're_uids'] = top_replies.re_uids.apply(lambda uid_lst: np.intersect1d(uid_lst, U))

	logger.info('5.6) Leave only the replies that have at least one replied uid in U')
	top_replies = top_replies.loc[top_replies.re_uids.str.len()>0].reset_index(drop=True)

	top_replies_time_span = pd.to_timedelta(top_replies.ts.max() - top_replies.ts.min())
	end_time = time.time() - start_time
	logger.info(f'''
	Top replies log stats
		> Number of posts: {top_replies.shape[0]}
		> Time Span: {top_replies_time_span}
		> 'top_replies' size: {get_mem_usage(object=top_replies)}
		> {get_run_time(seconds=end_time)}
	''')

	start_time = time.time()

	logger.info('5.7) Leave mentions which tweeting uid is present in U')
	top_mentions = MT.loc[MT.tw_uid.isin(U)].reset_index(drop=True)

	logger.info('5.8) Leave only the mentioned uids that are present in U')
	top_mentions.loc[:, 'mt_uids'] = top_mentions.loc[:, 'mt_uids'].apply(lambda uid_lst: [] if uid_lst is None else np.intersect1d(uid_lst, U))

	logger.info('5.9) Leave only the mentiones that have at least one replied uid in U')
	top_mentions = top_mentions.loc[top_mentions.mt_uids.str.len()>0].reset_index(drop=True)

	top_mentions_time_span = pd.to_timedelta(top_mentions.ts.max() - top_mentions.ts.min())
	end_time = time.time() - start_time
	logger.info(f'''
	> Top mentions log stats
		> Number of posts: {top_mentions.shape[0]}
		> Time Span: {top_mentions_time_span}
		> 'top_mentions' size: {get_mem_usage(object=top_mentions)}
		> {get_run_time(seconds=end_time)}
	''')

	start_time = time.time()
	top_tweets_time_span = pd.to_timedelta(top_tweets.ts.max() - top_tweets.ts.min())

	# 1. Add tweet ids and re-tweeted uids to lists
	top_tweets.loc[:, 'tw_id'] = top_tweets.tw_id.apply(lambda tw_id: [tw_id])
	# 2. Group by tweeting uid
	top_tweets_gb = top_tweets.groupby('tw_uid')['tw_id'].agg([('tw_ids', sum)])
	# 3. Flatten the lists
	top_tweets_gb.loc[:, 'tw_ids'] = top_tweets_gb.loc[:, 'tw_ids'].apply(lambda tw_ids: np.array([tw_id for tw_id in tw_ids if not np.isnan(tw_id)], dtype=np.int64).flatten())

	logger.info(f'''
	> Top Tweets log stats
		> Number of posts: {top_tweets.shape[0]}
		> Time Span: {top_tweets_time_span}
		> 'top_teweets' size: {get_mem_usage(object=top_tweets)}
		> {top_tweets.head()}
	''')

	logger.info(f'1.4) Load top retweets')
	top_retweets_time_span = pd.to_timedelta(top_retweets.ts.max() - top_retweets.ts.min())

	# 1. Add tweet ids and re-tweeted uids to lists
	top_retweets.loc[:, 'rt_uid'] = top_retweets.rt_uid.apply(lambda rt_uid: [rt_uid])
	# 2. Group by tweeting uid
	top_retweets_gb = top_retweets.groupby('tw_uid')['rt_uid'].agg([('rt_ed', sum)])
	# 3. Flatten the lists
	top_retweets_gb.loc[:, 'rt_ed'] = top_retweets_gb.loc[:, 'rt_ed'].apply(lambda uids: np.array([uid for uid in uids if not np.isnan(uid)], dtype=np.int64).flatten())

	logger.info(f'''
	> Top Re-tweets log stats
		> Number of posts: {top_retweets.shape[0]}
		> Time Span: {top_retweets_time_span}
		> 'top_retweets' size: {get_mem_usage(object=top_retweets)}
		> {top_retweets.head()}
	''')

	logger.info(f'1.5) Load top replies')
	top_replies_time_span = pd.to_timedelta(top_replies.ts.max() - top_replies.ts.min())

	# 1. Add tweet ids and re-tweeted uids to lists
	top_replies.loc[:, 're_uids'] = top_replies.loc[:, 're_uids'].apply(lambda uids: list(uids))
	# 2. Group by tweeting uid
	top_replies_gb = top_replies.groupby('tw_uid')['re_uids'].agg([('re_ed', sum)])
	# 3. Flatten the lists
	top_replies_gb.loc[:, 're_ed'] = top_replies_gb.loc[:, 're_ed'].apply(lambda uids: np.array([uid for uid in uids if not np.isnan(uid)], dtype=np.int64).flatten())

	logger.info(f'''
	> Top Replies log stats
		> Number of posts: {top_replies.shape[0]}
		> Time Span: {top_replies_time_span}
		> 'top_replies' size: {get_mem_usage(object=top_replies)}
		> {top_replies.head()}
	''')

	logger.info(f'1.6) Load top mentions')
	top_mentions_time_span = pd.to_timedelta(top_mentions.ts.max() - top_mentions.ts.min())

	# 1. Add tweet ids and re-tweeted uids to lists
	top_mentions.loc[:, 'mt_uids'] = top_mentions.loc[:, 'mt_uids'].apply(lambda uids: list(uids))

	# 2. Group by tweeting uid
	top_mentions_gb = top_mentions.groupby('tw_uid')['mt_uids'].agg([('mt_ed', sum)])

	# 3. Flatten the lists
	top_mentions_gb.loc[:, 'mt_ed'] = top_mentions_gb.loc[:, 'mt_ed'].apply(lambda uids: np.array([uid for uid in uids if not np.isnan(uid)], dtype=np.int64).flatten())

	logger.info(f'''
	> Top Mentions log stats
		> Number of posts: {top_mentions.shape[0]}
		> Time Span: {top_mentions_time_span}
		> 'top_mentions' size: {get_mem_usage(object=top_mentions)}
		> {top_mentions.head()}
	''')

	start_time = time.time()
	# 1. Join the groupbys
	top_social_network_df = top_tweets_gb.join(top_retweets_gb, how='outer')
	top_social_network_df = top_social_network_df.join(top_replies_gb, how='outer')
	top_social_network_df = top_social_network_df.join(top_mentions_gb, how='outer')

	# 2. Relevant UIDS
	R = top_social_network_df.index.values

	# 3. Filter tweets, replies and mentions that are not in R or the same as the uid of the user itself
	def _filter_irrelevant_uids(row, col, R):
		uids = row[col]
		relevant_uids = np.nan
		if not np.isnan(uids).all():
			relevant_uids = np.setdiff1d(np.intersect1d(uids, R), row.name)
			if relevant_uids.shape[0] == 0:
				relevant_uids = np.nan
		return relevant_uids

	top_social_network_df.loc[:, 'rt_ed'] = top_social_network_df.apply(_filter_irrelevant_uids, args=('rt_ed', R), axis=1)
	top_social_network_df.loc[:, 're_ed'] = top_social_network_df.apply(_filter_irrelevant_uids, args=('re_ed', R), axis=1)
	top_social_network_df.loc[:, 'mt_ed'] = top_social_network_df.apply(_filter_irrelevant_uids, args=('mt_ed', R), axis=1)

	# 4. Count the number of uids in R that the current uid reacted to (i.e. retweeted, replied or mentioned)
	def _sum_activities(row):
		n_tweets = 0 if np.isnan(row.tw_ids).all() else row.tw_ids.shape[0]
		n_retweeted = 0 if np.isnan(row.rt_ed).all() else row.rt_ed.shape[0]
		n_replied = 0 if np.isnan(row.re_ed).all() else row.re_ed.shape[0]
		n_mentioned = 0 if np.isnan(row.mt_ed).all() else row.mt_ed.shape[0]

		return n_tweets + n_retweeted + n_replied + n_mentioned

	top_social_network_df.loc[:, 'n_activities'] = top_social_network_df.apply(_sum_activities, axis=1)
	top_social_network_df.reset_index(inplace=True)
	end_time = time.time() - start_time

	logger.info(f'''
	> Social Network Stats
		> Number of top users: {top_social_network_df.shape[0]}
		> 'top_social_network_df' size: {get_mem_usage(object=top_social_network_df)}
		> {get_run_time(seconds=end_time)}
	''')

	check_dir(dir_path=save_dir_path, logger=logger)

	logger.info(f'1.7) Save the top social network at {TOP_SOCIAL_NETWORK_FILE}')
	top_social_network_df.to_pickle(TOP_SOCIAL_NETWORK_FILE)

	top_posts_file = save_dir_path / f'top_posts.pkl'
	logger.info(f'1.8) Save the top posts at {top_posts_file}')
	top_posts.to_pickle(TOP_POSTS_FILE)
	# top_posts.to_pickle(top_posts_file)

	top_tweets_file = save_dir_path / f'top_tweets.pkl'
	logger.info(f'1.9) Save the top tweets at {top_tweets_file}')
	top_tweets.to_pickle(top_tweets_file)

	top_retweets_file = save_dir_path / f'top_retweets.pkl'
	logger.info(f'1.10) Save the top retweets at {top_retweets_file}')
	top_retweets.to_pickle(top_retweets_file)

	top_replies_file = save_dir_path / f'top_replies.pkl'
	logger.info(f'1.11) Save the top replies at {top_replies_file}')
	top_replies.to_pickle(top_replies_file)

	top_mentions_file = save_dir_path / f'top_mentions.pkl'
	logger.info(f'1.12) Save the top mentions at {top_mentions_file}')
	top_mentions.to_pickle(top_mentions_file)

	I_file = save_dir_path / f'I.npy'
	logger.info(f'1.13) Save the I at {I_file}')
	np.save(I_file, I)

	U_file = save_dir_path / f'U.npy'
	logger.info(f'1.14) Save the U at {U_file}')
	np.save(U_file, U)

	end_time = time.time() - start_time
	logger.info(f'''
	Total Save Time: {get_run_time(seconds=end_time)}
	''')

	return top_social_network_df


def analyze_graph(social_network_df: pd.DataFrame, save_dir_path: pathlib.Path, logger: logging.Logger) -> None:

	def _get_mean_shortest_path(G: nx.Graph) -> None:

		t_sub_start = datetime.now()
		logger.info(f'Finding the mean shortest path...')
		E_L, exec_time = time_exec(func=partial(nx.average_shortest_path_length, G), logger=logger)
		logger.info(f'''
		Mean Shortest Path:
		 - E[L] = {E_L}
		 - Runtime: {exec_time}
		''')

	def _draw_graph(G: nx.Graph, node_degrees: dict, save_name: str) -> None:
		t_sub_start = datetime.now()

		logger.info(f'Plotting the largest connected component...')
		nx.draw(G, with_labels=False, nodelist=node_degrees.keys(), node_size=[v * 0.01 for v in node_degrees.values()])

		graphs_dir_path = save_dir_path / 'social_network_graphs'
		check_dir(dir_path=graphs_dir_path, logger=logger)

		graph_save_file_path = graphs_dir_path / save_name
		logger.info(f'Saving the largest connected component plot to \'{graph_save_file_path}\'...')
		plt.savefig(str(graph_save_file_path))

		logger.info(f'''
		 - The graph was saved to: {graph_save_file_path}
		 - Runtime: {datetime.now() - t_sub_start}
		''')

	def _plot_degree_dist(node_degrees: np.ndarray, dist_label: str, save_name: str) -> None:
		def _pow_law(x, k):
			return x**(-k)

		logger.info(f'Plotting the distribution of the node degrees...')

		t_sub_start = datetime.now()
		node_degrees_vals = np.array(list(node_degrees.values()))

		deg_hist, x_out = np.histogram(node_degrees_vals, range=(0, node_degrees_vals.max()), bins=node_degrees_vals.max() + 1, density=True)


		pow_law_2 = partial(_pow_law, k=2)
		pow_law_2_vals = np.apply_along_axis(pow_law_2, 0, x_out[:-1])
		pow_law_3 = partial(_pow_law, k=3)
		pow_law_3_vals = np.apply_along_axis(pow_law_3, 0, x_out[:-1])

		plt.style.use('ggplot')
		plt.figure(figsize=(15, 10))
		plt.bar(x_out[:-1], deg_hist, label=dist_label)
		plt.plot(x_out[:-1], pow_law_2_vals, label=u"p(d)=d\N{SUPERSCRIPT TWO}")
		plt.plot(x_out[:-1], pow_law_3_vals, label=u"p(d)=d\N{SUPERSCRIPT THREE}")
		plt.fill_between(x_out[:-1], pow_law_2_vals, pow_law_3_vals, alpha=0.5)
		plt.xlim([0, 40])
		plt.xlabel('d (node degree)', fontsize=30)
		plt.ylabel('p(d)', fontsize=30)
		plt.xticks(fontsize=30)
		plt.yticks(fontsize=30)
		plt.legend(fontsize=30)

		degree_dist_plots_dir_path = save_dir_path / 'degree_distribution_plots'
		check_dir(dir_path=degree_dist_plots_dir_path, logger=logger)

		degree_dist_plot_file_path = degree_dist_plots_dir_path / save_name
		plt.savefig(str(degree_dist_plot_file_path))

		logger.info(f'''
		 - The distribution plots were saved to: {degree_dist_plot_file_path}
		 - Runtime of 8): {datetime.now() - t_sub_start}
		''')

	t_start = datetime.now()

	logger.info(f'''
	---------------------
	=== analyze_graph ===
	---------------------
	''')

	logger.info(f'1) Build the undirected graph')
	t_sub_start = datetime.now()
	G = nx.Graph()
	edges = list(chain(*social_network_df.loc[:, 'edges'].values))
	G.add_edges_from(edges)
	logger.info(f'''
	 - Runtime of 1): {datetime.now() - t_sub_start}
	''')

	logger.info(f'2) Finding the largest connected component in an undirected graph')
	t_sub_start = datetime.now()
	largest_connected_component = max(nx.connected_components(G), key=len)
	edges_connected_component = [e for e in list(chain(*social_network_df.loc[:, 'edges'].values)) if e[0] in largest_connected_component]
	logger.info(f'''
	 - Runtime of 2): {datetime.now() - t_sub_start}
	''')

	logger.info(f'3) Build the largest connected graph from the general connected component')
	t_sub_start = datetime.now()
	G_connected_component = nx.Graph()
	G_connected_component.add_edges_from(edges_connected_component)
	logger.info(f'''
	 - Runtime of 3): {datetime.now() - t_sub_start}
	''')

	logger.info(f'4) Building the directed graph from the connected component')
	t_sub_start = datetime.now()
	dir_G_connected_component = nx.DiGraph()
	dir_G_connected_component.add_edges_from(edges_connected_component)
	logger.info(f'''
	 - Runtime of 4): {datetime.now() - t_sub_start}
	''')

	logger.info(f'5) Find the in/out degrees...')
	t_sub_start = datetime.now()
	in_degrees = dict(dir_G_connected_component.in_degree)
	out_degrees = dict(dir_G_connected_component.out_degree)
	logger.info(f'''
	 - Runtime of 5): {datetime.now() - t_sub_start}
	''')

	logger.info(f'6) Find the shortest mean path')
	_get_mean_shortest_path(G_connected_component)

	logger.info(f'7) Plot the connected components graphs')
	logger.info(f'7.1) Plot the in-degrees of the connected components graphs')
	# _draw_graph(G = G_connected_component, node_degrees=in_degrees, save_name='in_degrees_connected_component_plot.png')
	logger.info(f'7.2) Plot the out-degrees of the connected components graphs')
	# _draw_graph(G = G_connected_component, node_degrees=out_degrees, save_name='out_degrees_connected_component_plot.png')

	logger.info(f'8) Plot the degree distributions')
	# _plot_degree_dist(node_degrees=in_degrees, dist_label='in degrees', save_name='in_degree_distribution_of_the_largest_connected_component.png')
	# _plot_degree_dist(node_degrees=out_degrees, dist_label='out degrees', save_name='out_degree_distribution_of_the_largest_connected_component.png')

	logger.info(f'''
	Total Runtime for Graph Analysis: {datetime.now() - t_start}
	''')


def get_x_y(posts_df: pd.DataFrame, social_network_df: pd.DataFrame, activity_histogram_bins: int, save_dir_path: pathlib.Path, logger: logging.Logger = None) -> None:
	def _get_ts_range_vec(tw_log, soc_net, start_ts, end_ts):
		n_users = soc_net.index.shape[0]
		assert n_users > 0, f'The number of users in the network must be positive!'
		ts_rng_tw_log = tw_log[(tw_log.ts >= start_ts) & (tw_log.ts < end_ts)]
		act_uids_vec = None
		uids = act_cnts = 0
		# - Create zero vector
		act_uids_cnt_vec = np.zeros(shape=(soc_net.shape[0], ), dtype=np.int8)
		# IF THE ACTIVITY VECTOR IS NOT EMPTY
		if not ts_rng_tw_log.empty:

			# 1) Create activity counts vector
			# - Find the unique uids and the number of the activities each initiated in the time window
			ts_rng_uids, ts_rng_act_cnts = np.unique(ts_rng_tw_log.tw_uid.values, return_counts=True)
			# - Find which of the active users is present in the social network, and find his index
			relevant_uid_idx = ((ts_rng_uids[:, np.newaxis] == soc_net.tw_uid.values[np.newaxis, :]).sum(axis=1)>0)
			# print(relevant_uid_idx.shape)
			# - Find the uid of the relevant users
			ts_rng_uids = ts_rng_uids[relevant_uid_idx]
			# - Find the number of times this (relevant only) users were active in the time window
			ts_rng_act_cnts = ts_rng_act_cnts[relevant_uid_idx]
			# - Find the indices of the active users
			act_uids_idx = soc_net[soc_net.tw_uid.isin(ts_rng_uids)].index
			# - Poulate the activity count vector with the number of times each user was active
			act_uids_cnt_vec[act_uids_idx] = ts_rng_act_cnts

			# 2) Create activity binary vector
			act_uids_vec = np.zeros(shape=(soc_net.index.shape[0], ), dtype=np.int8)
			# - Find active indices
			act_uids_idx = soc_net[soc_net.tw_uid.isin(ts_rng_tw_log.tw_uid.values)].index
			# - Mark active indices
			act_uids_vec[act_uids_idx] = np.int8(1)


		return act_uids_vec, uids, act_uids_cnt_vec

	def _plot_activity_histogram(data1: dict, data2: dict, layout_args: dict, save_dir: pathlib.Path, save_name: str = 'activity_distribution'):
		plt.style.use(layout_args['style'])
		fig, ax = plt.subplots(figsize=layout_args['fig_size'])

		# 4) Plot the X histogram
		ax.bar(
			x=data1['x'],
			height=data1['bars'],
			width=layout_args['bar_width'],
			color=layout_args['bars1']['bar_color'],
			label=layout_args['bars1']['bar_label']
		)

		ax.bar(
			x=data2['x'],
			height=data2['bars'],
			width=layout_args['bar_width'],
			color=layout_args['bars2']['bar_color'],
			label=layout_args['bars2']['bar_label']
		)

		# 5) Plot the legend
		if layout_args['legend']['show']:
			plt.legend(loc=layout_args['legend']['location'], fontsize=layout_args['legend']['font_size'])

		# 7) Configure the layout
		ax.set(
			title=layout_args['title'],
			xticks=layout_args['xticks']['values'],
			yticks=layout_args['yticks']['values'],
		)
		plt.rc('xtick', labelsize=layout_args['xticks']['font_size'])	# fontsize of the tick labels
		plt.rc('ytick', labelsize=layout_args['yticks']['font_size'])	# fontsize of the tick labels
		ax.set_xlabel(layout_args['xlabel']['text'], fontsize=layout_args['xlabel']['font_size'])
		ax.set_ylabel(layout_args['ylabel']['text'], fontsize=layout_args['ylabel']['font_size'])

		check_dir(dir_path=save_dir, logger=logger)
		plt.savefig(save_dir/f"{save_name}.png")
		plt.close(fig)

	logger.info(f'''
	---------------
	=== get_x_y ===
	---------------
	''')
	X_tw_log = posts_df
	y_tw_log = posts_df[posts_df.loc[:, 'type'].isin(['Re-Tweet', 'Mention', 'Reply'])]

	start_ts = X_tw_log.ts.min()
	end_ts = X_tw_log.ts.max()

	# MAIN LOOP
	X_rng_lst = []
	X_uid_lst = []
	X_cnt_lst = []

	y_rng_lst = []
	y_uid_lst = []
	y_cnt_lst = []

	X_total_act = np.zeros(social_network_df.shape[0])
	y_total_act = np.zeros(social_network_df.shape[0])
	while start_ts + TIME_WINDOW < end_ts:

		X_vec, X_uids, X_act_cnts_vec = _get_ts_range_vec(tw_log=X_tw_log, soc_net=social_network_df, start_ts=start_ts, end_ts=start_ts + TIME_WINDOW)
		X_total_act += X_act_cnts_vec

		y_vec, y_uids, y_act_cnts_vec = _get_ts_range_vec(tw_log=y_tw_log, soc_net=social_network_df, start_ts=start_ts + TIME_WINDOW, end_ts=start_ts + 2*TIME_WINDOW)
		y_total_act += y_act_cnts_vec

		if X_vec is None or y_vec is None:
			print(f'<!> WARRNING in ExpData::get_X_y_vecs(): The train/test activity vector(s) for the time range [{start_ts} : {start_ts + TIME_WINDOW}] are empty!')
		else:
			X_rng_lst.append(X_vec)
			X_uid_lst.append(X_uids)
			X_cnt_lst.append(X_act_cnts_vec)

			y_rng_lst.append(y_vec)
			y_uid_lst.append(y_uids)
			y_cnt_lst.append(y_act_cnts_vec)
		start_ts += TIME_WINDOW

	X = np.array(X_rng_lst)
	X_uids = np.array(X_uid_lst, dtype=object)
	X_cnts = np.array(X_cnt_lst, dtype=object)

	y = np.array(y_rng_lst)
	y_uids = np.array(y_uid_lst, dtype=object)
	y_cnts = np.array(y_cnt_lst, dtype=object)

	check_dir(dir_path=save_dir_path, logger=logger)

	X_file =  save_dir_path / f'X.npy'
	y_file =  save_dir_path / f'y.npy'

	np.save(X_file, X)
	np.save(y_file, y)

	logger.info(f'''
	Activity Vectors Stats:
		> Time window: {TIME_WINDOW}
		> Number of activity vectors: {X.shape[0]}
		> X:
			- 1's Proportion: {X.mean():.3f}+/-{X.std():.3f} %
			- Shape: {X.shape}
			- Size: {get_mem_usage(object=X)}
		> y:
			- 1's Proportion: {y.mean():.3f}+/-{y.std():.3f} %
			- Shape: {y.shape}
			- Size: {get_mem_usage(object=y)}
	''')

	# 1) Create a directory for plots
	act_hist_plots_dir = save_dir_path / 'plots'
	check_dir(dir_path=act_hist_plots_dir, logger=logger)

	X_relative_act = X_total_act / X.shape[0]
	Y_relative_act = y_total_act / y.shape[0]
	bars1, x1 = np.histogram(X_relative_act, bins=np.arange(activity_histogram_bins))
	bars2, x2 = np.histogram(Y_relative_act, bins=np.arange(activity_histogram_bins))

	max_bar = np.max(np.concatenate([bars1, bars2]))
	step = np.power(10, int(np.log10(max_bar)))

	logger.info(f'''
	Relative Activity in X:
	    {bars1}
	X Bins:
	    {x1+.25}
	---
	Relative Activity in y:
	    {bars2}
	y Bins:
	    {x2+.75}
	''')
	_plot_activity_histogram(
		data1=dict(
			x=x1[:-1]+.25,
			bars=bars1
		),
		data2=dict(
			x=x2[:-1]+.75,
			bars=bars2
		),
		layout_args=dict(
			fig_size=(20, 10),
			bar_width=.5,
			bars1=dict(
				bar_color='navy',
				bar_label='Tweets',
			),
			bars2=dict(
				bar_color='green',
				bar_label='Responses',
			),
			style='ggplot',
			legend=dict(show=True, location='upper right', font_size=35) ,
			title='',#'User Activity Histogram',
			xlabel=dict(text='Number of Posts', font_size=35),
			ylabel=dict(text='Number of Users', font_size=35),
			xticks=dict(values=np.arange(0, activity_histogram_bins, 1), font_size=30),
			yticks=dict(values=np.arange(0, max_bar, step), font_size=30)
		),
		save_dir=act_hist_plots_dir,
		save_name=f'{NET_NAME}_activity_histogram',
	)


def get_A(social_network_df: pd.DataFrame, save_dir_path: pathlib.Path, adjacency_type: str = 'all', logger: logging.Logger = None) -> None:
	'''
	:adjacency_type: which uids are considered to be neighbors, one of 'all' (i.e. 'rt_ed', 're_ed' and 'mt_ed'), 're_ed', 're_ed' or 'mt_ed'
	'''

	logger.info(f'''
	-------------
	=== get_W ===
	-------------
	''')
	start_time = time.time()
	# 1. Get the number of users in the social network
	N = social_network_df.index.shape[0]

	# 2. Initialize an empty adjacency matrix
	A = np.zeros((N, N), dtype=np.int8)


	# 4. Mark with '1' each neighbor of the user
	for uid_idx in social_network_df.index:
		if adjacency_type=='all' or adjacency_type=='rt_ed':
			rt_ed = social_network_df.loc[uid_idx, 'rt_ed']
			if not np.isnan(rt_ed).all():
				rt_ed_idxs = social_network_df.loc[social_network_df.tw_uid.isin(rt_ed)].index.values
				A[uid_idx, rt_ed_idxs] = np.int8(1)

		if adjacency_type=='all' or adjacency_type=='re_ed':
			re_ed = social_network_df.loc[uid_idx, 're_ed']
			if not np.isnan(re_ed).all():
				re_ed_idxs = social_network_df.loc[social_network_df.tw_uid.isin(re_ed)].index.values
				A[uid_idx, re_ed_idxs] = np.int8(1)

		if adjacency_type=='all' or adjacency_type=='mt_ed':
			mt_ed = social_network_df.loc[uid_idx, 'mt_ed']
			if not np.isnan(mt_ed).all():
				mt_ed_idxs = social_network_df.loc[social_network_df.tw_uid.isin(mt_ed)].index.values
				A[uid_idx, mt_ed_idxs] = np.int8(1)

	end_time = time.time() - start_time
	logger.info(f'''
	> A stats
		> Proportion of neighbors: {A.sum()} / {A.shape[0]**2} ({100*A.sum() / A.shape[0]**2:.3f}%)
		> A's shape: {A.shape}
		> A's size: {get_mem_usage(object=A)}
		> E[A]: {A.mean():.4f}+/-{A.std():.4f}
		> A sum: {A.sum()}
		> |A| sum: {np.abs(A).sum()}
		> {get_run_time(seconds=end_time)}
	''')
	check_dir(dir_path=save_dir_path, logger=logger)

	A_file =  save_dir_path / f'A.npy'

	start_time = time.time()

	np.save(A_file, A)

	end_time = time.time() - start_time
	logger.info(get_run_time(seconds=end_time))


def get_social_network_stats(social_network_df: pd.DataFrame, save_dir_path: pathlib.Path, logger: logging.Logger) -> None:
	logger.info(f'''
	--------------------------------
	=== get_social_network_stats ===
	--------------------------------
	''')
	logger.info('1) Get In / Out Edges')
	social_network_df.loc[:, 'edges'] = social_network_df.loc[:, 'tw_uid'].apply(lambda uid: [(uid, f) for f in np.hstack(social_network_df.loc[social_network_df.loc[:,'tw_uid']==uid, ['rt_ed', 're_ed', 'mt_ed']].values[0]) if ~np.isnan(f)])

	logger.info('2) Analyze graph')
	analyze_graph(social_network_df=social_network_df, save_dir_path=save_dir_path, logger=logger)


def load_data_from_file(data_file_path: pathlib.Path, logger: logging.Logger):
	data = None
	load_time = None
	if isinstance(data_file_path, pathlib.Path) and data_file_path.is_file():
		logger.info(f'Trying to loading the data from {data_file_path}...')
		dir_path, file_name, file_type = get_dir_path_file_name_file_type(path=data_file_path, logger=logger)
		if file_type == 'pkl' or file_type == 'pickle':
			data, load_time  = time_exec(func=partial(pd.read_pickle, str(data_file_path)), logger=logger)
		elif file_type == 'parq':
			data, load_time  = time_exec(func=partial(pd.read_parquet, str(data_file_path)), logger=logger)
	return data, load_time


if __name__=='__main__':

	logger = get_logger(configs_file_path = LOGGER_CONFIGS_FILE_PATH)
	logger.info(f'1) Create logger')

	logger.info(f'2) Configure the clean data directory, which will contain the checkpoints')
	check_dir(dir_path=INPUT_PKL_DIR, logger=logger)

	logger.info(f'3) Load the DataFrame')
	logger.info(f'3.1) Try loading the clean data from {POSTS_DF_PKL_FILE_PATH} ...')
	posts_df, _ = load_data_from_file(data_file_path=POSTS_DF_PKL_FILE_PATH, logger=logger)
	if posts_df is None:
		logger.info(f'3.2) Try loading the original data from {POSTS_DF_PARQ_FILE_PATH} ...')
		posts_df, _= time_exec(func=partial(get_clean_posts_df, posts_df_parq_file_path=POSTS_DF_PARQ_FILE_PATH, save_dir_path=INPUT_PKL_DIR, logger=logger), logger=logger)

		if posts_df is None:
			logger.error(f'<X> File does not exist: \'{posts_df_parq_file_path}\'')
			if logger is not None:
				logger.exeption(
					f'No data file was found at: {posts_df_parq_file_path}'
				)
			sys.exit(1)

	logger.info(
	f'''
	Log Summary
		Total number of posts: {posts_df.shape[0]}
	'''
	)
	logger.info(f'4) Top Social Network')
	if isinstance(TOP_SOCIAL_NETWORK_FILE, pathlib.Path) and TOP_SOCIAL_NETWORK_FILE.is_file():
		logger.info(f'4.1) Try loading the social network data frame from \'{TOP_SOCIAL_NETWORK_FILE}\' ...')
		top_social_network_df, _ = load_data_from_file(data_file_path=TOP_SOCIAL_NETWORK_FILE, logger=logger)
	else:
		logger.info(f'4.2) Could not load the social network data frame from \'{TOP_SOCIAL_NETWORK_FILE}\' (the file does not exist at this location)...')
		logger.info(f'4.3) Extracting the \'top_social_network_df\' from \'posts_df\'...')
		top_social_network_df, _= time_exec(func=partial(get_top_social_network_df, posts_df=posts_df, save_dir_path=INPUT_PKL_DIR, logger=logger), logger=logger)

	if top_social_network_df.shape[0] > N:
		idx = np.random.choice(np.arange(top_social_network_df.shape[0]), N, replace=False)
		top_social_network_df = top_social_network_df.loc[idx].reset_index(drop=True)

	logger.info(f'5) Get Social Network Stats')
	check_dir(dir_path=SOCIAL_NETWORK_DIR, logger=logger)
	time_exec(func=partial(get_social_network_stats, social_network_df=top_social_network_df, save_dir_path=SOCIAL_NETWORK_DIR, logger=logger), logger=logger)

	logger.info(f'6) Get the adjacency matrix A from tweet log')
	check_dir(dir_path=VECTORS_DIR, logger=logger)
	time_exec(func=partial(get_A, social_network_df=top_social_network_df, save_dir_path=VECTORS_DIR, adjacency_type='all', logger=logger), logger=logger)

	logger.info(f'7) Get the activity vectors X and y')
	top_posts_df, _ = load_data_from_file(data_file_path=TOP_POSTS_FILE, logger=logger)
	time_exec(func=partial(get_x_y, posts_df=top_posts_df, social_network_df=top_social_network_df, activity_histogram_bins=5, save_dir_path=VECTORS_DIR, logger=logger), logger=logger)
