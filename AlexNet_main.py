import torch.optim as optim
import torch.utils.data
from torchvision import transforms as transforms
import numpy as np
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import cohen_kappa_score
from data_loader_channels import MyDataset
 
from AlexNet import AlexNet 

CLASSES = ('0', '1')

parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epoch', default=1000, type=int, help='number of epochs tp train for')
args = parser.parse_args()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        self.train_loader = MyDataset(datatxt='Data/train_expand/data_list.txt', transform=transforms.ToTensor())
        self.test_loader = MyDataset(datatxt='Data/verify_expand/data_list.txt', transform=transforms.ToTensor())
    
    def load_model(self):
        self.device = torch.device('cpu')
        self.model = AlexNet(num_classes=2)  

        # Adagrad
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.device) 

    #训练过程
    def train(self):
        print("train:", end='')
        self.model.train()  
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):  
            data= data.view(1,11,80,80)  
            target=torch.tensor([target])
            self.optimizer.zero_grad()

            # forward
            output = self.model(data)  
            loss = self.criterion(output, target) 

            # backward
            loss.backward() 
            self.optimizer.step() 

            train_loss += loss.item()  
            prediction = torch.max(output, 1)  

            total += target.size(0) 

            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy()) 

        print('Loss: %.4f | Acc: %.3f%% (%d/%d)' % (
            train_loss / len(self.train_loader), 100. * train_correct / total, train_correct, total))
       
        train_loss1=train_loss / len(self.train_loader) 
        return train_loss1, train_correct / total  

    def test(self):
        print("test:", end='')
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader): 
                
                data= data.view(1,11,80,80) 
                target=torch.tensor([target])
                output = self.model(data) 
                loss = self.criterion(output, target)
                test_loss += loss.item() 
                prediction = torch.max(output, 1) 
                total += target.size(0) 
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy()) 

            print('Loss: %.4f | Acc: %.3f%% (%d/%d)' % (
                test_loss / len(self.test_loader), 100. * test_correct / total, test_correct, total))
        test_loss1=test_loss / len(self.test_loader)
        return test_loss1, test_correct / total

    def test_with_select_model(self, model_name):  
        print("test select model:")
        model = torch.load(model_name) 
        
        model.eval() 
        self.load_data()
        test_loader=self.test_loader
        
        correct = 0
        total = 0
        TN=0
        TP=0
        FP=0
        FN=0
        predict_value = []
        target_value = []
        probability_value=[]
        
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(test_loader): 
               
                
                data= data.view(1,11,80,80) 
                target=torch.tensor([target])
                output = model(data)

                probability = nn.functional.softmax(output,dim=1)
                prediction = torch.max(output, 1) 
                
                total += target.size(0)
                if (target.cpu().numpy()==0):
                    if (prediction[1].cpu().numpy()==0):
                         TN = TN+1
                    
                if (target.cpu().numpy()==0):
                     if (prediction[1].cpu().numpy()==1):
                         FP = FP+1
                    
                if (target.cpu().numpy()==1):
                     if (prediction[1].cpu().numpy()==0):
                         FN = FN+1
                    
                if (target.cpu().numpy()==1):
                    if (prediction[1].cpu().numpy()==1):
                         TP = TP+1
                
                correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())#正确的个数
                
                pred=prediction[1].cpu().numpy() 
                predict_value.append(pred[0])
                
                targ=target.cpu().numpy()
                target_value.append(targ[0])
                
                
                probab=probability.cpu().numpy()
                probability_value.append(probab[:,1])
                
        
        print('ACC :%.3f %d/%d' % (correct / total * 100, correct, total))
        print('TN:',TN)
        print('FP:',FP)
        print('FN:',FN)
        print('TP:',TP)
               
        
        
        kappa = cohen_kappa_score(np.array(target_value).reshape(-1,1), np.array(predict_value).reshape(-1,1))
        print('Kappa：',kappa)
        
        fpr, tpr, thresholds = roc_curve(target_value, probability_value, drop_intermediate=False)

        AUC = auc(fpr, tpr)
        print("AUC : ", AUC)
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='CNN(area = {:.3f})'.format(AUC))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
                

    def run(self, save_fig=True):
        self.load_data() 
        self.load_model() 
        accuracy = 0

        loss_list_train = []
        loss_list_test = []
        accuracy_train=[]
        accuracy_test=[]
        accuracy_train_list=[]
        accuracy_test_list=[]
        
        max_test=0
        epoch_best=0
        for epoch in range(1, self.epochs + 1):  
            print("\n===> epoch: %d/%d" % (epoch, self.epochs))

            train_result = self.train()  
            loss_list_train.append(train_result[0])
            accuracy_train.append(train_result[1])

            test_result = self.test()  

            accuracy = max(accuracy, test_result[1])
            accuracy_test.append(test_result[1])
            loss_list_test.append(test_result[0])
            
            if ( test_result[1]>=max_test):
                max_test=test_result[1]
                model_path = "model_best.pth"
                torch.save(self.model, model_path)
                epoch_best=epoch
        
            if (epoch%100== 0):
                 a=str(epoch)    
                 model_out_path = "model_" +a+ ".pth"
                 torch.save(self.model, model_out_path)
                 print("Checkpoint saved to {}".format(model_out_path))
                 accuracy_train_list.append(train_result[1])
                 accuracy_test_list.append(test_result[1])

                 
        
if __name__ == '__main__':
    solver = Solver(args)
    solver.run(save_fig=True)  
    solver.test_with_select_model("model_1000.pth") 
    
    

    

