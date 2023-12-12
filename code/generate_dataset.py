import pandas as pd
import numpy as np
import pickle

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class Mydataset(Dataset):
    def __init__(self,data_list,label_list):
        self.data_list = data_list
        self.label_list = label_list
    def __getitem__(self,index):
        return self.data_list[index],self.label_list[index]
    def __len__(self):
        return len(self.data_list)



df = pd.read_excel('/home/ruixing/workspace/eeg_liujie/train.xlsx')
data_list = np.array(df.iloc[:, 2:].values)
label_o = np.array(df.iloc[:, 1].values)
# label_list = [0 if i == 'neg' else 1 for i in label_o]

train_data, test_data, train_labels, test_labels = train_test_split(
                                                    data_list, 
                                                    label_o, 
                                                    test_size=0.2, 
                                                    random_state=43)

train_dataset = Mydataset(train_data,train_labels)
test_dataset = Mydataset(test_data,test_labels)

with open('/home/ruixing/workspace/eeg_liujie/dataset/train_dataset.pkl','wb') as file:
    pickle.dump(train_dataset,file)

with open('/home/ruixing/workspace/eeg_liujie/dataset/test_dataset.pkl','wb') as file:
    pickle.dump(test_dataset,file)

