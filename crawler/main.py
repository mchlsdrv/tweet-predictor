import os
import time
import yaml
import pandas as pd
import numpy as np
import threading
import logging
import concurrent.futures
from pathlib import (
    Path
)

from aux_functions import (
    crwl_sub_net,
    get_logger,
    check_dir,
    get_run_time,
    get_ts,
)

if __name__ == '__main__':
    start_time = time.time()

    # 1) Get the configs
    configs_dir_path = Path('./configs')
    configs_file_path = configs_dir_path / 'system_configs.yml'
    assert configs_file_path.is_file(), f'No configurations file was found at \'{configs_file_path}\''

    with configs_file_path.open(mode='r') as conf_file:
        configs = yaml.safe_load(conf_file.read())

    # 1.1) Get the app credentials
    app_credentials = configs.get('app_credentials')
    assert app_credentials is not None, f'No app credentials were found!'
    n_workers = len(app_credentials)

    # 1.2) Get general configs
    general_configs = configs.get('general')
    network_name = general_configs.get('network_name')

    logger = None
    logger_configs_file_path = Path(general_configs['logger_configs_file'])
    if logger_configs_file_path.is_file():
        logger = get_logger(configs_file_path=logger_configs_file_path)

    output_dir_path = Path(general_configs['output_dir'])
    if not output_dir_path.is_dir():
        os.makedirs(output_dir_path)
    crawled_df_file_path = output_dir_path / f'{network_name}_crawled_df.pkl'

    # 1.3) Get crawler specific configs
    crawler_configs = configs.get('crawler')

    temp_dir_path = Path('./temp')
    if not temp_dir_path.is_dir():
        os.makedirs(temp_dir_path)

    # 2) Get the uncrawled uids
    uids_file_path = Path(crawler_configs.get('uids_file'))
    assert uids_file_path.is_file(), f'No file containing the UIDs were found at \'./{uids_file_path}\''

    uncrawled_uids = np.load(uids_file_path)

    # 3.2) Get the UID splits for each thread

    # 3) Assign jobs to threads
    # 3.3) Run the thread pool
    lock = threading.Lock()

    threads = list()


    for thr_idx, uids_sub_arr, credentials in zip(np.arange(n_workers), np.array_split(uncrawled_uids, n_workers), np.array(list(app_credentials.values()))[:n_workers]):

        logging.info(f'{__name__}: create and start thread {thr_idx}')
        x = threading.Thread(
            target=crwl_sub_net,
            args=(
                f'{network_name}_{thr_idx}',
                uids_sub_arr,
                credentials,
                crawled_df_file_path,
                lock,
                logger
            )
        )
        threads.append(x)
        x.start()

    for idx, thread in enumerate(threads):
        logging.info(f'{__name__}: before joining thread {idx}')
        thread.join()
        logging.info(f'{__name__}: after joining thread {idx}')

    if isinstance(logger, logging.Logger):
        logger.info(f'''
IN: {__name__}
    Total Run Time:
        {get_run_time(time.time() - start_time)}
        ''')
