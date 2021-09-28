import os
import yaml
import logging.config
import threading
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
from classes import TwitterCrawler


def configure_output_dir(dir_path: pathlib.Path, default_dir_path: pathlib.Path, logger: logging.Logger = None) -> pathlib.Path:
    # Ensure that the log dir exists
    assert isinstance(dir_path, pathlib.Path) and isinstance(default_dir_path, pathlib.Path), f'Both \'dir_path\' and \'default_dir_path\' parameters must be of type \'pathlib.Path\', but of types \'{type(dir_path)}\' and \'{type(default_dir_path)}\' respectively!'

    log_dir_path = dir_path
    try:
        if not log_dir_path.is_dir():
            os.makedirs(log_dir_path)
    except Exception as err:
        if isinstance(logger, logging.Logger):
            logger.exception(f'Could not create the \'{log_dir_path}\' due to error: {err}!')
            logger.exception(f'Default log directory (\'{default_dir_path}\') will be used to store logs!')
        log_dir_path = default_dir_path
        if not log_dir_path.is_dir():
            os.makedirs(log_dir_path)
    return log_dir_path


def get_logger(configs_file_path: pathlib.Path) -> logging.Logger:
    logger = None

    with configs_file_path.open(mode='r') as f:
        configs = yaml.safe_load(f.read())
        # Check that the log directory is valid
        dir_path, file_name, file_type = get_dir_path_file_name_file_type(configs['handlers']['logfile']['filename'])
        valid_log_dir_path = configure_output_dir(dir_path=dir_path, default_dir_path=Path('./logs'), logger=None)
        # Assign a valid path to the log file
        configs['handlers']['logfile']['filename'] = str(valid_log_dir_path / f'{file_name}.{file_type}')
        logging.config.dictConfig(configs)
        logger = logging.getLogger(__name__)

    return logger

def get_dir_path_file_name_file_type(path: pathlib.Path, logger: logging.Logger = None) -> (pathlib.Path, str, str):
    dir_path = None
    file_name_type = None
    file_name = None
    file_type = None

    if isinstance(logger, logging.Logger):
        logger.info('''
----------------------------------------
=== get_dir_path_file_name_file_type ===
----------------------------------------
    ''')
        logger.info(f'Trying to parse the \'{path}\' path ...')

    path_str = str(path)
    try:
        # Try the windows file notation
        dir_path = Path(path_str[::-1][path_str[::-1].index('\\')+1:][::-1])
        file_name_type = path_str[::-1][:path_str[::-1].index('\\')][::-1]
    except ValueError as err:
        if isinstance(logger, logging.Logger):
            logger.exception(err)
            logger.info(f'Trying to parse the \'{path}\' path in a Linux style ...')
        # If the file is not represented in windows style - try parsing it in linux notation
        try:
            dir_path = Path(path_str[::-1][path_str[::-1].index('/')+1:][::-1])
            file_name_type = path_str[::-1][:path_str[::-1].index('/')][::-1]
        except ValueError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)
                logger.info(f'Could not parse the \'{path}\' path !')
    if isinstance(logger, logging.Logger):
        logger.info(f'Successfully parsed the \'{path}\' path into {file_name_type} !')

    if file_name_type is not None:
        file_name, file_type = file_name_type[:file_name_type.index('.')], file_name_type[file_name_type.index('.')+1:]

    if isinstance(logger, logging.Logger):
        logger.info(f'''
The path \'{path}\' was successfully parsed into:
    - Directory path: {dir_name}
    - File name: {file_name}
    - File type: {file_type}
    ''')

    return dir_path, file_name, file_type


def get_ts() -> str:
    def _clean_ts(ts):
        ts = ts[::-1]
        ts = ts[ts.index('.')+1:]
        ts = ts[::-1]
        ts = ts.replace(':', '_')
        return ts
    ts = str(datetime.now())
    return _clean_ts(ts)


def check_dir(dir_path: pathlib.Path, logger: logging.Logger = None) -> bool:
    dir_ok = False
    if isinstance(dir_path, Path) or isinstance(dir_path, WindowsPath):
        if not dir_path.is_dir():
            makedirs(dir_path)
        dir_ok = True
        if isinstance(logger, logging.Logger):
            logger.info(f'The \'{dir_path}\' is valid.')
    else:
        if isinstance(logger, logging.Logger):
            logger.info(f'ERROR in check_dir: The path to save_dir ({dir_path}) is not of type \'Path\' but of type {type(dir_path)}!')
    return dir_ok


def get_run_time(seconds: int) -> str:
	hours = int(seconds // 3600)
	minutes = int((seconds - hours * 3600) // 60)
	residual_seconds = int(seconds - hours * 3600 - minutes * 60)
	return f'Runtime\n\t=> {hours}:{minutes}:{residual_seconds} [H:m:s]'


# FUNCTIONS
def crwl_sub_net(network_name: str, uids: np.ndarray, credentials: dict, save_file_path: pathlib.Path, lock: threading.Lock, logger: logging.Logger = None) -> None:
    # 1) Create the crawler
    crwlr = TwitterCrawler(
        credentials=credentials
    )

    if isinstance(logger, logging.Logger):
        with lock:
            logger.info(
            f'''
IN: {__name__}
Thread with ID {threading.get_ident()} Started:
    > Credentials: {credentials}
            ''')
    # 2) Crawl the network
    new_crawled_df = crwlr.crawl_network(uids=uids, network_name=network_name, lock=lock, logger=logger)

    # 3) Update the crawled database
    with lock:
        if save_file_path.is_file():
            crawled_df = pd.read_pickle(save_file_path)
            crawled_df = crawled_df.append(new_crawled_df, ignore_index=True)
            crawled_df.to_pickle(save_file_path, protocol=4)
        else:
            new_crawled_df.to_pickle(save_file_path, protocol=4)
