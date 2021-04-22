

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('-s', '--sge', type=str, default='nosge')
	parser.add_argument('-l', '--filelist', type=str, default='')
	parser.add_argument('-zr', '--zr_root', type=str, default='/home/korhan/Desktop/zerospeech2017/track2/src/ZRTools')

	args = parser.parse_args()