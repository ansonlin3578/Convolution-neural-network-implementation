import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pdb
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
import gc
import os
from datetime import datetime

## wrap train dataset
class Train_set(Dataset):
    def __init__(self, img_arr, label_arr):
        self.img_arr = img_arr
        self.label_arr = label_arr

    def __len__(self):
        return(len(self.img_arr))

    def __getitem__(self, index):
        img = self.img_arr[index]
        if torch.from_numpy(img) is not None:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
        if self.label_arr is not None:
            label = self.label_arr[index]
            return img, label
        else:
            return img

class Val_set(Dataset):
    def __init__(self, img_arr, label_arr):
        self.img_arr = img_arr
        self.label_arr = label_arr

    def __len__(self):
        return(len(self.img_arr))

    def __getitem__(self, index):
        img = self.img_arr[index]
        if torch.from_numpy(img) is not None:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
        if self.label_arr is not None:
            label = self.label_arr[index]
            return img, label
        else:
            return img

## wrap test dataset
class Test_set(Dataset):
    def __init__(self, img_arr, label_arr):
        self.img_arr = img_arr
        self.label_arr = label_arr

    def __len__(self):
        return(len(self.img_arr))

    def __getitem__(self, index):
        img = self.img_arr[index]
        if torch.from_numpy(img) is not None:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
        if self.label_arr is not None:
            label = self.label_arr[index]
            return img, label
        else:
            return img

## Design your own cnn model
class My_Model(nn.Module):
    def __init__(self):
        super(My_Model, self).__init__()
        
        # Defining a 2D convolution layer
        self.conv_1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1, dilation=1) #in 64 out 64
        self.bn_1 = nn.BatchNorm2d(12)
        self.relu_1 = nn.ReLU(inplace=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=4, padding=0, dilation=1)#in 64 out 16
        # Defining another 2D convolution layer
        self.conv_2 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, dilation=1)#in 16 out 16
        self.bn_2 = nn.BatchNorm2d(12)
        self.relu_2 = nn.ReLU(inplace=True)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)#in 16 out 8
        
        self.FC = nn.Linear(12*8*8, 6)

    def forward(self, input):
        out = self.conv_1(input)
        out = self.bn_1(out)
        out = self.relu_1(out)
        out = self.maxpool_1(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu_2(out)
        out = self.maxpool_2(out)
        out = out.view(out.size(0), -1)
        out = self.FC(out)
        return out


def cal_accuracy(prediction, label):
    ''' Calculate Accuracy, please don't modify this part
        Args:
            prediction (with dimension N): Predicted Value
            label  (with dimension N): Label
        Returns:
            accuracy:　Accuracy
    '''

    accuracy = 0
    number_of_data = len(prediction)
    for i in range(number_of_data):
        accuracy += float(prediction[i] == label[i])
    accuracy = (accuracy / number_of_data) * 100

    return accuracy

def main():
    with h5py.File(r"C:\Users\User\Desktop\碩一上修課資料\深度學習_林嘉文\HW3\dataset\Signs_Data_Training.h5", "r") as train_f:
        print("Keys: %s" %train_f.keys())
        for key in train_f.keys():
            print(train_f[key].shape)
        train_data = np.array(list(train_f["train_set_x"]))
        train_data = train_data / 255.0
        train_label = np.array(list(train_f["train_set_y"]))
        # print(type(train_data[0]))
        # print(type(train_data))

    with h5py.File(r"C:\Users\User\Desktop\碩一上修課資料\深度學習_林嘉文\HW3\dataset\Signs_Data_Testing.h5", "r") as test_f:
        print("Keys: %s" %test_f.keys())
        for key in test_f.keys():
            print(test_f[key].shape)
        test_data = np.array(list(test_f["test_set_x"]))
        test_data = test_data / 255.0
        test_label = np.array(list(test_f["test_set_y"]))
        # print(type(test_data[0]))
        # print(type(test_label[0]))

    #################################### K fold ############################################
    n_epochs = 20
    batch_size = 4
    num_folds = 5
    kfold = KFold(n_splits = num_folds, shuffle=True)
    resnet_swith = False
    # K-fold Cross Validation model evaluation
    fold_no = 1 

    test_set = Test_set(test_data, test_label)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    total_train_loss_plot, total_train_acc_plot = [] , []
    total_val_loss_plot , total_val_acc_plot = [] , []
    def reset_weights(model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    for train_idx, val_idx in kfold.split(train_data, train_label):
        ###################################create model#####################################
        # defining the model
        model = My_Model()
        
        # defining the loss function
        criterion = nn.CrossEntropyLoss()
        if resnet_swith:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            in_ch = model.fc.in_features
            model.fc = nn.Linear(in_ch, 6)

        # checking if GPU is available
        if torch.cuda.is_available():
            model = model.cuda()
            model.double()
            criterion = criterion.cuda()

        # defining the optimizer
        # optimizer = optim.Adam(model.parameters(), lr=0.005)
        optimizer = optim.Adam(model.parameters())
        # print(model)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        train_set = Train_set(train_data[train_idx], train_label[train_idx])
        train_loader = DataLoader(train_set, batch_size=batch_size)
        val_set = Train_set(train_data[val_idx], train_label[val_idx])
        val_loader = DataLoader(val_set, batch_size=batch_size)

        # empty list to store training losses
        train_loss_plot , train_acc_plot = [], []
        val_loss_plot, val_acc_plot = [] , []
        
        ######################################## training the model #######################################
        for epoch in range(n_epochs):
            train_loss_per_batch = []
            train_acc_per_batch = []
            # getting the training set
            for data, label in train_loader:
                if torch.cuda.is_available():
                    x_train = data.cuda()
                    y_train = label.cuda()
                
                # prediction for training and validation set
                output_train = model(x_train)

                # computing the training loss
                loss_train = criterion(output_train, y_train)
                # clearing the Gradients of the model parameters
                optimizer.zero_grad()

                train_loss_per_batch.append(loss_train.cpu())
                # computing the updated weights of all the model parameters
                loss_train.backward()
                optimizer.step()

                # prediction for training set
                # softmax = torch.exp(output_train).cpu()
                # prob = list(softmax.detach().numpy())

                to_pred = output_train.cpu().detach().numpy()
                predictions = np.argmax(to_pred, axis=1)
                # accuracy on training set
                g_truth = y_train.cpu().detach().numpy()
                train_acc_per_batch.append(accuracy_score(g_truth, predictions))

            train_losses_per_epoch = np.sum(train_loss_per_batch) / (len(train_loss_per_batch))
            train_acc_per_epoch = np.sum(train_acc_per_batch) / (len(train_acc_per_batch))
            train_loss_plot.append(train_losses_per_epoch)
            train_acc_plot.append(train_acc_per_epoch)
            
            ######################################### validation #####################################
            val_loss_per_batch = []
            val_acc_per_batch = []
            with torch.no_grad():
                for data, label in val_loader:
                    if torch.cuda.is_available():
                        x_val = data.cuda()
                        y_val = label.cuda()

                    output_val = model(x_val)

                    # computing the validation loss
                    loss_val = criterion(output_val, y_val)
                    val_loss_per_batch.append(loss_val.cpu())

                    # prediction for validation set
                    # softmax = torch.exp(output_val).cpu()
                    # prob = list(softmax.detach().numpy())
                    
                    to_pred = output_val.cpu().detach().numpy()
                    predictions = np.argmax(to_pred, axis=1)
                    # accuracy on validation set
                    g_truth = y_val.cpu().detach().numpy()
                    val_acc_per_batch.append(accuracy_score(g_truth, predictions))

            val_losses_per_epoch = np.sum(val_loss_per_batch) / (len(val_loss_per_batch))
            val_acc_per_epoch = np.sum(val_acc_per_batch) / (len(val_acc_per_batch))
            val_loss_plot.append(val_losses_per_epoch)
            val_acc_plot.append(val_acc_per_epoch)
            
            # if epoch%2 == 0:
            print('Epoch : ',epoch+1, '\t', 'train loss :', train_losses_per_epoch, '\t' , "train acc :", train_acc_per_epoch)
            print('Epoch : ',epoch+1, '\t', 'val loss :', val_losses_per_epoch, '\t' , "val acc :", val_acc_per_epoch)
            # if epoch == 4:
            #     pdb.set_trace()

        # Saving the model
        if resnet_swith:
            save_path = f'./resnet-fold-{fold_no}-epoch-{n_epochs}.pth'
            torch.save(model.state_dict(), save_path)
        else:
            save_path = f'./model-fold-{fold_no}-epoch-{n_epochs}.pth'
            torch.save(model.state_dict(), save_path)

        model.apply(reset_weights)
        total_train_loss_plot.append(train_loss_plot)
        total_train_acc_plot.append(train_acc_plot)
        total_val_loss_plot.append(val_loss_plot)       
        total_val_acc_plot.append(val_acc_plot)     #val_acc_plot is all epochs accuarcy in one fold
        
        if resnet_swith:
            directory = "resnet"
            parent_dir = "C:/Users/User/Desktop/碩一上修課資料/深度學習_林嘉文/HW3"
            path = os.path.join(parent_dir, directory)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "resnet-fold-{}_epoch-{}Info.txt".format(fold_no, n_epochs)), 'a') as fp:
                fp.write("time-{} | Train loss. {:.4f} | Train acc {:.4f} | Val loss. {:.4f} | Val acc {:.4f}\n".format(
                    datetime.now().strftime("%M:%D:%H:%M:%S"),
                    train_loss_plot[-1],
                    train_acc_plot[-1],
                    val_loss_plot[-1],
                    val_acc_plot[-1],
                ))
                fp.close()
        else:
            directory = "origianl"
            parent_dir = "C:/Users/User/Desktop/碩一上修課資料/深度學習_林嘉文/HW3"
            path = os.path.join(parent_dir, directory)
            # if not os.path.exists(path):
            #     os.mkdir(path)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "fold-{}_epoch-{}Info.txt".format(fold_no, n_epochs)), 'a') as fp:
                fp.write("time-{} | Train loss. {:.4f} | Train acc {:.4f} | Val loss. {:.4f} | Val acc {:.4f}\n".format(
                    datetime.now().strftime("%M:%D:%H:%M:%S"),
                    train_loss_plot[-1],
                    train_acc_plot[-1],
                    val_loss_plot[-1],
                    val_acc_plot[-1],
                ))
                fp.close()

        # Increase fold number
        fold_no = fold_no + 1
        
        #clean gpu
        # torch.cuda.empty_cache()
        # del model
        # del criterion
        # gc.collect()
    #################################### choose_model ####################################
    total_acc_avg_per_fold = []
    for i in range(len(total_val_acc_plot)):        #total folds
        avg_per_fold = np.sum(total_val_acc_plot[i]) / len(val_acc_plot) #calulate avg acc in total epochs
        total_acc_avg_per_fold.append(avg_per_fold)
    max_idx = total_acc_avg_per_fold.index(max(total_acc_avg_per_fold))  #get the highest acc model in folds

    best_model = My_Model()
    if resnet_swith:
        best_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        in_ch = best_model.fc.in_features
        best_model.fc = nn.Linear(in_ch, 6)
        
        best_model.load_state_dict(torch.load(f'./resnet-fold-{max_idx + 1}-epoch-{n_epochs}.pth'))   #load the best model
    else:
        best_model.load_state_dict(torch.load(f'./model-fold-{max_idx + 1}-epoch-{n_epochs}.pth'))   #load the best model
    best_model.eval()
    best_model = best_model.cuda()
    best_model.double()
    if resnet_swith:
        print("Resnet best model is the fold-{}-epoch-{}".format(max_idx + 1, n_epochs))
    else:
        print("the best model is the fold-{}-epoch-{}".format(max_idx + 1, n_epochs))
    #################################### test_pred ####################################
    test_pred = []
    total_truth = []
    with torch.no_grad():
        for data, label in test_loader:
            x_test, y_test = Variable(data), Variable(label)
            if torch.cuda.is_available():
                x_test = x_test.cuda()
                y_test = y_test.cuda()

            output_test = best_model(x_test)

            # prediction for test set
            softmax = torch.exp(output_test).cpu()
            prob = list(softmax.detach().numpy())
            predictions = np.argmax(prob, axis=1)
            test_pred.extend(predictions)

            g_truth = y_test.cpu().detach().numpy()
            total_truth.extend(g_truth)

    ###################### final prediction ######################
    prediction = test_pred# predict the first data is class 5, and second is class 0 ... 
    label = total_truth # first GT is class 5, and second is class 1 ... 
    Testing_accuracy = cal_accuracy(prediction, label)
    
    pic_name = ''
    epoch_len = len(total_train_loss_plot[0])
    x_axis = np.array([i+1 for i in range(epoch_len)])
    for i in range(len(total_train_loss_plot)):
        plt.subplot(2, 3, i+1)
        plt.plot(x_axis , total_train_loss_plot[i])
        plt.plot(x_axis , total_val_loss_plot[i])
        pic_name = "train_loss_{}epochs".format(n_epochs)
        plt.xlabel("Epoch")
        plt.ylim(0, 1.4)
        plt.ylabel("Loss")
        plt.title("Loss plot-epoch-{}".format(n_epochs))
    if resnet_swith:
        plt.savefig("resnet-"+pic_name)
    else:
        plt.savefig(pic_name)
    plt.close()

    for i in range(len(total_train_acc_plot)):
        plt.subplot(2, 3, i+1)
        plt.plot(x_axis , total_train_acc_plot[i])
        plt.plot(x_axis , total_val_acc_plot[i])
        pic_name = "train_acc_{}epochs".format(n_epochs)
        plt.xlabel("Epoch")
        plt.ylim(0, 1.2)
        plt.ylabel("accuracy")
        plt.title("Accuracy plot-epoch-{}".format(n_epochs))
    if resnet_swith:
        plt.savefig("resnet-"+pic_name)
    else:
        plt.savefig(pic_name)
    plt.close()
    return Testing_accuracy

if __name__=='__main__':
    Testing_accuracy = main()
    print('Testing_accuracy(%):', Testing_accuracy)

