from data_loader.data_loader import *

data_loader = FamilyMartDataLoader('../data', 1, False, True, 0, 1)
print(len(data_loader))
for batch_idx, (data, target1, target2) in enumerate(data_loader):
	print(batch_idx, data[:10])