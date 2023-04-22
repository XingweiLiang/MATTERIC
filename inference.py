import argparse
import os
import sys

import numpy as np
import pandas as pd
import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from models.si_lstm import SILSTM
from utils import load_parameters
from dataloader import IEMOCAPDataset, MELDDataset

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

iemocap_map = {'happy':0, 'sad':1, 'neural':2, 'angry':3, 'excited ':4, ' frustrated':5}
meld_map = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}


def main(args):
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # 加载数据
    upper_db = args.dataset.upper()
    path = args.feature_path + f'{upper_db}_features/{upper_db}_features_raw.pkl'
    roberta_path = args.feature_path + f'{upper_db}_features/{upper_db}_features_roberta.pkl'
    if args.dataset == 'IEMOCAP':
        testset = IEMOCAPDataset(path=path, roberta_path=roberta_path, train=False)
    else:
        testset = MELDDataset(path=path, roberta_path=roberta_path, n_classes=7, train=False)
    test_loader = DataLoader(testset,
                             batch_size=1,
                             collate_fn=testset.collate_fn,
                             num_workers=0)

    # 初始化模型
    n_classes = 6 if args.dataset == 'IEMOCAP' else 7
    model = SILSTM(n_classes, args.dataset)
    load_parameters(model, args.model_checkpoint_path)
    model.to(device)
    model.eval()

    # 验证
    preds, labels, masks, ids = [], [], [], []

    for data in tqdm.tqdm(test_loader):
        if args.dataset == 'IEMOCAP':
            r1, r2, r3, r4, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-1]]
        else:
            r1, r2, r3, r4, textf, acouf, qmask, umask, label = [d.to(device) for d in data[:-1]]
        
        textf = (r1 + r2 + r3 + r4) / 4
        lp_, x_a, x_l = model(torch.cat((textf,acouf), dim=-1), qmask, umask)

        labels_ = label.view(-1) # batch*seq_len
        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        ids.extend(data[-1][0])
        masks.append(umask.view(-1).cpu().numpy())
    
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)

    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)

    print(f"Acc {avg_accuracy:.2f}, Fscore {avg_fscore:.2f}")

    map_ = iemocap_map if args.dataset == 'IEMOCAP' else meld_map
    lab2emo = {v: k for k, v in map_.items()}

    df = pd.DataFrame({'ids': ids, 
                       'labels': list(map(lambda x: lab2emo[x], labels)), 
                       'preds': list(map(lambda x: lab2emo[x], preds))})
    df.to_csv(args.output_dir + f'/{args.dataset}_results.csv', index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument('--dataset', type=str, default='IEMOCAP', help='IEMOCAP / MELD')
    parser.add_argument('--feature_path', type=str, default='./features/')
    parser.add_argument("-M", "--model_checkpoint_path", type=str, default='exps/silstm/IEMOCAP/best_model.model')
    parser.add_argument("-O", "--output_dir", type=str, default='./case')
    parser.add_argument("--gpu", type=int, default=3)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
