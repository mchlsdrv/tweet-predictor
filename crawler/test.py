import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import threading
import concurrent.futures
import time


def thread_function(name, rnd_num):
    print(f'{__name__}Thread {name}: starting')
    time.sleep(2)
    print(f'{__name__}Thread {name} : thread id {threading.get_ident()}, {rnd_num}')
    time.sleep(1)
    print(f'{__name__}Thread {name}: finishing')


if __name__=='__main__':
    configs_file_path = Path('D:\\[University]\\[M.Sc.]\\[Thesis]\\[Experiments]\\[Code]\\[Python]\\[Crawler]\\configs\\system_configs.yml')
    with configs_file_path.open(mode='r') as f:
        configs = yaml.safe_load(f.read())
        for config in configs:
            sub_configs = configs.get(config)
            if isinstance(sub_configs, dict):
                for sub_config in sub_configs:
                    print(sub_configs[sub_config])
            else:
                print(sub_configs)
    list(configs.get('twitter_apps').values())

    list(zip(configs.get('twitter_apps').values(), np.arange(10)))
    fmt = '%(asctime)s: %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO, datefmt='%H:%M:%S')

    threads = list()

    for idx in range(3):
        logging.info(f'{__name__}: create and start thread {idx}')
        x = threading.Thread(target=thread_function, args=(idx, np.random.randint(0, 10)))
        threads.append(x)
        x.start()

    for idx, thread in enumerate(threads):
        logging.info(f'{__name__}: before joining thread {idx}')
        thread.join()
        logging.info(f'{__name__}: after joining thread {idx}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(thread_function, list(zip(np.arange(10), np.random.randint(0, 10, 10))))

    list(zip([1, 2, 3], [1]*3, [5]*3))
    apps = configs.get('twitter_apps')
    len(configs.get('twitter_apps'))
    d = dict(a=1, b=2)
    d['a']
    a = np.arange(99)
    np.array_split(a, 10)
    a.size
    a[1000:]
    idx = np.linspace(0, 100, 9).astype(np.int16)
    for idx_start, idx_end in zip(idx[:-1], idx[1:]):
        print(a[idx_start:idx_end])
    list(zip(idx[:-1], idx[1:]))[0]
    a[0:12]
    a[idxs]

    A = pd.DataFrame(dict(a=[1, 2, 3], b=[3, 4, 5]))
    B = pd.DataFrame(dict(a=[1, 2, 3], b=[3, 4, 5]))
    A = A.append(B, ignore_index=True)
    A

    k = dict(zip(np.arange(100), np.random.randint(0, 10, 100)))
    np.array(list(k.values()))[:10]
