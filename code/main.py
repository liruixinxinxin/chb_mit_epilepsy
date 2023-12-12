from generate_dataset import Mydataset
from torch.utils.data import DataLoader
from network import *
from parameters import *
from train import *
import pickle



# load data
with open('/home/ruixing/workspace/eeg_liujie/dataset/train_dataset.pkl','rb') as file:
    train_dataset = pickle.load(file)
with open('/home/ruixing/workspace/eeg_liujie/dataset/test_dataset.pkl','rb') as file:
    test_dataset = pickle.load(file)

train_dataloader  = DataLoader(train_dataset,batch_size=batch_size,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset))


model = ann


ann_train(device=device,
          train_dataloader=train_dataloader,
          test_dataloader = test_dataloader,
          model=ann)