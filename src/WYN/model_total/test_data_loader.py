from data_loader.data_loader import *

if __name__ == "__main__":
	data_loader = FamilyMartDataLoader('data', 1, False, False, 0, 1)
	# print(len(data_loader))
	for batch_idx, (data, target1) in enumerate(data_loader):
		print(batch_idx, data[:10])