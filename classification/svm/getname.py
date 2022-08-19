import json
from dataset import HSICity2
from torch.utils.data import DataLoader

test_set = HSICity2("/data/huangyx/data/HSICityV2/test", use_hsi=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

ret = {}

for i, (image, label, hsi, name) in enumerate(test_loader):
    ret[name[0]] = i

with open('result-mapping.json', 'w') as f:
    json.dump(ret, f, indent=2)
