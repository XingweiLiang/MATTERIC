import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd

from loss import MaskedLoss, InfoNCE
from models.si_lstm import SILSTM


class ModelTrainer(nn.Module):

    def __init__(self, device, lr, test_step, lr_decay, loss, n_classes, dataset, model, fusion_type, mult_task, **kwargs):

        super(ModelTrainer, self).__init__()
        self.device = device
        self.dataset = dataset
        self.mult_task = mult_task
        # 定义模型
        if model == 'SILSTM':
            self.model = SILSTM(n_classes, dataset, fusion_type, mult_task).to(self.device)
 
        # 定义损失函数
        if loss == 'CrossEntropy':
            losser = nn.CrossEntropyLoss
        elif loss == 'NLL':
            losser = nn.NLLLoss
        self.loss = MaskedLoss(losser).to(self.device)
        self.infoNCELoss = InfoNCE(negative_mode='unpaired')

        # 定义优化器
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)

        # 打印模型参数大小
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2fM" % (
                sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        # Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        lr = self.optim.param_groups[0]['lr']
        losses, masks = [], []

        for num, data in enumerate(loader):
            self.optim.zero_grad()  # 梯度置为0
            # import ipdb;ipdb.set_trace()
            if self.dataset == 'IEMOCAP':
                r1, r2, r3, r4, visuf, acouf, qmask, umask, label = [d.to(self.device) for d in data[:-1]]
            else:
                r1, r2, r3, r4, textf, acouf, qmask, umask, label = [d.to(self.device) for d in data[:-1]]
            
            textf = (r1 + r2 + r3 + r4) / 4
            output = self.model(torch.cat((textf,acouf), dim=-1), qmask, umask)

            labels_ = label.view(-1) # batch*seq_len
            
            loss = 0
            for m in self.mult_task:
                loss += self.loss(output[m], labels_, umask)

            masks.append(umask.view(-1).cpu().numpy())
            losses.append(loss.item()*masks[-1].sum())
            loss.backward()
            self.optim.step()

        masks = np.concatenate(masks)
        avg_loss = round(np.sum(losses)/np.sum(masks), 4)

        return lr, avg_loss

    def eval_network(self, loader):
        self.eval()

        # 预测结果，实际结果
        preds, labels, masks = [], [], []
        l_preds, a_preds = [], []

        with torch.no_grad():
            for num, data in enumerate(loader):
                if self.dataset == 'IEMOCAP':
                    r1, r2, r3, r4, visuf, acouf, qmask, umask, label = [d.to(self.device) for d in data[:-1]]
                else:
                    r1, r2, r3, r4, textf, acouf, qmask, umask, label = [d.to(self.device) for d in data[:-1]]
                
                textf = (r1 + r2 + r3 + r4) / 4
                output = self.model(torch.cat((textf, acouf), dim=-1), qmask, umask)

                labels_ = label.view(-1) # batch*seq_len
                for m in self.mult_task:
                    if m == 'M':
                        preds.append(torch.argmax(output['M'], 1).data.cpu().numpy())
                    elif m == 'A':
                        a_preds.append(torch.argmax(output['A'], 1).data.cpu().numpy())
                    elif m == 'T':
                        l_preds.append(torch.argmax(output['T'], 1).data.cpu().numpy())

                labels.append(labels_.data.cpu().numpy())
                masks.append(umask.view(-1).cpu().numpy())
        
        for m in self.mult_task:
            if m == 'M':
                preds = np.concatenate(preds)
            elif m == 'A':
                a_preds = np.concatenate(a_preds)
            elif m == 'T':
                l_preds = np.concatenate(l_preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)

        # df = pd.DataFrame({'pred': preds, 'labels': labels, 'masks': masks})
        # df.to_csv('res.csv', index=False)
        res = {
            'M': self.metrics(labels, preds, masks),
            'A': self.metrics(labels, a_preds, masks) if 'A' in self.mult_task else (0, 0),
            'T': self.metrics(labels, l_preds, masks) if 'T' in self.mult_task else (0, 0),
        }

        return res

    def metrics(self, labels, preds, weight):
        accuracy = round(accuracy_score(labels, preds, sample_weight=weight)*100, 2)
        avg_fscore = round(f1_score(labels, preds, sample_weight=weight, average='weighted')*100, 2)

        return accuracy, avg_fscore

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)