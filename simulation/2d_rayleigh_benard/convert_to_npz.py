import h5py
import numpy as np
import argparse
import glob


def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-f', '--file_pattern', required=True, type=str,
	                    help='File pattern of h5 files.')
	parser.add_argument('-o', '--outfile', type=str, default='snapshots.npz',
	                    help='Ouput file name.')

	args = parser.parse_args()
	files = sorted(glob.glob(args.file_pattern))

	scales = ['write_number', 'sim_time']
	variables = ['p', 'b', 'u', 'w', 'bz', 'uz', 'wz']

	# var_dict = dict(zip(variables + scales, [[]] * (len(variables) + 2)))
	var_dict = {}
	
	for key in (variables + scales):
		var_dict[key] = []

	for file in files:
		fh = h5py.File(file, 'r')
		for s in scales:
			var_dict[s].append(np.array(fh['scales'][s]))
		for v in variables:
			var_dict[v].append(np.array(fh['tasks'][v]))

	for key in var_dict.keys():
		var_dict[key] = np.concatenate(var_dict[key], axis=0)

	# sort based on write_number
	sort_idx = np.argsort(var_dict['write_number'])
	for key in var_dict.keys():
		var_dict[key] = var_dict[key][sort_idx]
		
	np.savez(args.outfile, **var_dict)

if __name__ == '__main__':
	main()