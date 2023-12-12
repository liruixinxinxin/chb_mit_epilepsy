import torch
num_class = 2
batch_size = 64
num_epochs = 100
time_step = 1
num_channel = 60

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')