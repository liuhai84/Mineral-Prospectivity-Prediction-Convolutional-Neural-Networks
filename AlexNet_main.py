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
from plotcm import plot_confusion_matrix
 
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
        self.test_loader1 = None
        self.predict_loader= None


    '''按照80%和20%分好的数据'''
    def load_data(self):
        self.train_loader = MyDataset(datatxt='Data1/train/data_list.txt', transform=transforms.ToTensor())
        self.test_loader = MyDataset(datatxt='Data1/verify/data_list.txt', transform=transforms.ToTensor())
        self.predict_loader = MyDataset(datatxt='Data1/predict/data_list.txt', transform=transforms.ToTensor())    
        self.test_loader1 = MyDataset(datatxt='Data1/test/data_list.txt', transform=transforms.ToTensor())

#特征重要性的输入F1~F11
#        self.predict_loader = MyDataset(datatxt='DataImportant1/F1/data_list.txt', transform=transforms.ToTensor())
    
    def load_model(self):
        self.device = torch.device('cpu')
        self.model = AlexNet(num_classes=2)  #model是AlexNet

        # Adagrad
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
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

    #测试
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

    def test_with_select_model(self, model_name, c):  
        print("test select model:")
        model = torch.load(model_name) #加载模型的名称
        
        model.eval() 
        self.load_data()
        
        if (c==0):
             test_loader=self.test_loader
        if (c==1):
             test_loader=self.test_loader1
        
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
            for batch_num, (data, target) in enumerate(test_loader): #加载数据
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
                
        result = np.array([target_value,predict_value, probability_value]).T
        if(c==0):
           np.savetxt(r"结果\验证.txt", result, delimiter = "\t")
        if(c==1):
           np.savetxt(r"结果\测试.txt", result, delimiter = "\t") 
        
        print('ACC :%.3f %d/%d' % (correct / total * 100, correct, total))
        print('TN:',TN)
        print('FP:',FP)
        print('FN:',FN)
        print('TP:',TP)
               
        names = ('non-d' ,'d') 
        plt.figure(figsize=(2,2))
        
        '''输出CM'''
        cm=[[TN,FP],[FN,TP]]
        cm=np.array(cm)
        plot_confusion_matrix(cm, names)  
        
        '''Sensitivity'''
        Sensitivity=TP/(TP+FN)
        print('Sensitivity:',Sensitivity)
        
        '''Specificity'''
        Specificity=TN/(TN+FP)
        print('Specificity:',Specificity)
        
        '''Positive predictive value'''
        if ((TP+FP)==0):
            Ppr=0
        else:
            Ppr=TP/(TP+FP)
        print('Positive predictive value:',Ppr)
        
        '''Negative predictive value'''
        if ((TN+FN)==0):
            Npr=0
        else:
            Npr=TN/(TN+FN)
        print('Negative predictive value:',Npr)
        
        '''Accuracy'''
        Accuracy=(TP+TN)/(TP+TN+FP+FN)
        print('Accuracy:',Accuracy*100,'%')
        
        '''F1'''
        F1=2*TP/(2*TP+FP+FN)
        print('F1:',F1)
        

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

        kappa = cohen_kappa_score(np.array(target_value).reshape(-1,1), np.array(predict_value).reshape(-1,1))
        print('Kappa：',kappa)
                

    #运行
    def run(self, save_fig=True):
        self.load_data() #加载训练数据和测试数据
        self.load_model() #加载模型
        accuracy = 0

        loss_list_train = []
        loss_list_test = []
        accuracy_train=[]#全部
        accuracy_test=[]#全部
        accuracy_train_list=[]#每100次保存一次
        accuracy_test_list=[]
        
        max_test=0
        epoch_best=0
        for epoch in range(1, self.epochs + 1):  #epochs=50
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/%d" % (epoch, self.epochs))

            train_result = self.train()  #迭代训练结果
            loss_list_train.append(train_result[0])
            accuracy_train.append(train_result[1])

            test_result = self.test()  #test结果

            accuracy = max(accuracy, test_result[1]) #最高准确率
            accuracy_test.append(test_result[1])
            loss_list_test.append(test_result[0])
            
            if (test_result[1]>=max_test):
                max_test=test_result[1]
                model_path = "model_best.pth"
                torch.save(self.model, model_path)
                epoch_best=epoch
        
            #每100次保存一次模型
            if (epoch%100== 0):
                 a=str(epoch)    
                 model_out_path = "model_" +a+ ".pth"
                 torch.save(self.model, model_out_path)
                 print("Checkpoint saved to {}".format(model_out_path))
                 accuracy_train_list.append(train_result[1])
                 accuracy_test_list.append(test_result[1])
            
            np.savetxt(r"结果\losstrain.txt", loss_list_train, delimiter = "\t")
            np.savetxt(r"结果\losstest.txt",  loss_list_test, delimiter = "\t")
            np.savetxt(r"结果\trainACC.txt",  accuracy_train_list, delimiter = "\t")
            np.savetxt(r"结果\testACC.txt",  accuracy_test_list, delimiter = "\t")
            np.savetxt(r"结果\Accuracy_train.txt",  accuracy_train, delimiter = "\t")
            np.savetxt(r"结果\Accuracy_test.txt",  accuracy_test, delimiter = "\t")
        
        print('最佳模型迭代次数：',epoch_best)
                 
            
    '''预测样本'''
    def predict_with_select_model(self, model_name):
        model = torch.load(model_name) 
        model.eval() 
        if self.predict_loader is None:
            self.load_data()
        predict_value = []
        probability_value=[]
        target_value=[]
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.predict_loader): #加载数据
                data, target = data.view(1, 11, 80,80), torch.tensor([target])

                output = model(data)
                probability = nn.functional.softmax(output,1)
                prediction = torch.max(output, 1) 
                
                pred=prediction[1].cpu().numpy() 
                predict_value.append(pred[0]) 
                
                
                probab=probability.cpu().numpy()
                probability_value.append(probab[:,1])
                
                targ=target.cpu().numpy()
                target_value.append(targ[0]) 
        
        result = np.array([predict_value, probability_value,target_value]).T
        np.savetxt(r"结果\预测.txt", result, delimiter = "\t")

    
      

if __name__ == '__main__':
    solver = Solver(args)
    solver.run(save_fig=True) #训练
    
    print('------20%验证------')
    solver.test_with_select_model("model_1000.pth",c=0)  #20%验证 
    
    print('------测试------')
    solver.test_with_select_model("model_1000.pth",c=1)  #测试 
    
    print('------预测------')
    solver.predict_with_select_model("model_1000.pth")  #预测
    

