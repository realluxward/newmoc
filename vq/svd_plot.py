import torch
from matplotlib import pyplot as plt
from IPython import embed
dataset = f"amazon-toys-games-filter"
split_dataset = f"toys-split"
train_iterations = 100
bs = 4096
device = "cuda:0"
shuffle = True
preprocess = False

avg_s=torch.load(f"./feat/{split_dataset}/avg_s.pt")
avg_rs_s = torch.load(f"./feat/{split_dataset}/avg_rs_s.pt")
avg_s = avg_s.cpu().numpy()
avg_rs_s = avg_rs_s.cpu().numpy()
s = avg_s/avg_s.max()
rs_s = avg_rs_s/avg_rs_s.max()
embed()
print((s>0.1).sum())
print((rs_s>0.1).sum())

plt.figure()
plt.plot(s)
plt.savefig("./sigma.pdf")
print(s.sum()/s.max())

plt.figure()
plt.plot(rs_s)
plt.savefig("./rs_sigma.pdf")
print(rs_s.sum()/rs_s.max())
