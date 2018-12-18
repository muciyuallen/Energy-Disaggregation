import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import datetime
import time
import math
import warnings
warnings.filterwarnings("ignore")
import glob

import sys
def main(house_number, test_only):
	labels = read_label()

	# print(labels)
	# sys.exit()
	df = read_merge_data(house_number, labels)
	dates = [str(time)[:10] for time in df.index.values]
	dates = sorted(list(set(dates)))
	print('House {0} data contain {1} days from {2} to {3}.'.format(house_number,len(dates),dates[0], dates[-1]))
	print(dates, '\n')

	des_path = "train_test_data/"

	if test_only:
		file = des_path+'H%s_test_full.csv'%house_number
		df.to_csv(file)
		print("%s saved to %s"%(file, des_path))
	else:
		df1_train = df.ix[:dates[10]]
		df1_val = df.ix[dates[11]:dates[16]]
		df1_test = df.ix[dates[17]:]
		print('df_train.shape: ', df1_train.shape)
		print('df_val.shape: ', df1_val.shape)
		print('df_test.shape: ', df1_test.shape)

		df1_train.to_csv(des_path+'H%s_train.csv'%house_number)
		df1_val.to_csv(des_path+'H%s_val.csv'%house_number)
		df1_test.to_csv(des_path+'H%s_test.csv'%house_number)
		print("files saved to ",des_path)

def read_label():
	label = {}
	for i in range(1, 7):
		low = 'Data/low_freq/house_{}/labels.dat'.format(i)
		label[i] = {}
		with open (low) as f:
			for line in f:
				splitted_line = line.split(' ')
				label[i][int(splitted_line[0])] = splitted_line[1].strip() + '_' + splitted_line[0]
	return label

def read_merge_data(house, labels):
	path = 'Data/low_freq/house_{}/'.format(house)
	file = path + 'channel_1.dat'
	df = pd.read_table(file, sep = ' ', names = ['unix_time', labels[house][1]], 
									   dtype = {'unix_time': 'int64', labels[house][1]:'float64'}) 
	
	num_apps = len(glob.glob(path + 'channel*'))
	for i in range(2, num_apps + 1):
		file = path + 'channel_{}.dat'.format(i)
		data = pd.read_table(file, sep = ' ', names = ['unix_time', labels[house][i]], 
									   dtype = {'unix_time': 'int64', labels[house][i]:'float64'})
		df = pd.merge(df, data, how = 'inner', on = 'unix_time')
	df['timestamp'] = df['unix_time'].astype("datetime64[s]")
	df = df.set_index(df['timestamp'].values)
	df.drop(['unix_time','timestamp'], axis=1, inplace=True)
	return df

if __name__ == '__main__':
	house_number = sys.argv[1]
	house_number = int(house_number)
	test_only = sys.argv[2]
	if test_only == 'y' or test_only == 'Y':
		test_only = True
	else:
		test_only = False
	main(house_number, test_only)



