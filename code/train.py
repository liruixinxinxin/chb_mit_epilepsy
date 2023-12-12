import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from parameters import *

def ann_train(device,train_dataloader,test_dataloader,model):
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    model.to(device)
    # opt = optim.Adam(model.parameters(), lr=0.000172)
    opt = optim.Adam(model.parameters(), lr=0.000172, weight_decay=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        true_labels = []
        predicted_labels = []
        for inputs, labels in tqdm(train_dataloader):
            inputs = (inputs.reshape(-1,1,time_step,num_channel)).to(device)
            labels = labels.to(device)
            opt.zero_grad()
            outputs = (model(inputs)).to(device)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            loss.backward()
            opt.step()
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())
            correct_predictions += torch.sum(predicted == labels).item()
        cm = confusion_matrix(true_labels, predicted_labels)
        train_accuracy = correct_predictions / len(predicted_labels)
        print("Train Accuracy:", train_accuracy)
        print("Train Confusion Matrix:")
        print(cm)

        test_loss = 0.0
        test_correct_predictions = 0
        model.eval()
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.reshape(-1,1,time_step,num_channel).to(device)
                # inputs[inputs == -1] = 2
                labels = labels.to(device)
                outputs = model(inputs).to(device)
                # outputs = torch.sum(outputs,dim=1)
                labels = labels.to(torch.long)
                loss = criterion(outputs, labels).to(device)
                _, predicted = torch.max(outputs, 1)
                # _, label = torch.max(labels, 1)
                test_loss += loss.item()
                test_correct_predictions += torch.sum(predicted == labels).item()
                true_labels.extend(labels.tolist())
                predicted_labels.extend(predicted.tolist())
        cm = confusion_matrix(true_labels, predicted_labels)
        print("Confusion Matrix:")
        print(cm)
        train_loss = total_loss / len(train_dataloader)
        
        test_loss /= len(test_dataloader)
        test_accuracy = test_correct_predictions / len(test_dataloader.dataset)
        print("Epoch", epoch+1)
        # print("Train Set:")
        # print("Loss:", train_loss)
        
        print("Test Set:")
        print("Loss:", test_loss)
        print(f"Accuracy:{test_accuracy}")
        model.eval()
        