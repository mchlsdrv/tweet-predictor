import pathlib
from pathlib import (
	Path
)
import pandas as pa
from pandas import (
	read_pickle as pkl_2_df
)


if __name__=='__main__':
	root_dir = Path('./[Temp]')
	for idx, pkl_file in enumerate(root_dir.iterdir()):
		try:
			df = pkl_2_df(pkl_file)
			# print(df.head())
		except Exception as e:
			print(f'Error occured for: {pkl_file}\n{e}\n---\n')
		else:
			print(f'Susscessfully loaded: {pkl_file}')
