import argparse
import time
import warnings
import sys

from numpy import argmax
import numpy as np
import torch

from model_trainer import ModelTrainer
from dataloader import get_loaders
from utils import init_args, seed_everything

warnings.simplefilter("ignore")


def main(args):
    """
    模型训练主函数

    Parameter:
        args:参数配置项
    """
    # 设置 device
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
 
    # 固定随机种子
    seed_everything(args.seed)
    
    # 加载数据
    train_loader, valid_loader, test_loader = get_loaders(
                                            args.feature_path,
                                            dataset=args.dataset,
                                            valid=0.2,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            n_classes=args.n_classes)

    # 记录参数
    score_file = open(args.score_save_path, "a+")
    score_file.write(f'Model: {args.model} \n'
                     f'Dataset : {args.dataset}\n'
                     f'Task : {args.mult_task} \n'
                     + '-' * 20 + '\n')

    # 模型测试
    if args.eval:
        s = ModelTrainer(device, **vars(args))
        print("Model %s loaded from previous state!" % args.initial_model)
        s.load_parameters(args.initial_model)
        res = s.eval_network(test_loader)
        print(f"M_Acc {res['M'][0]:.2f}, M_Fscore {res['M'][1]:.2f}, " \
              f"A_Acc {res['A'][0]:.2f}, M_Fscore {res['A'][1]:.2f}, " \
              f"T_Acc {res['T'][0]:.2f}, M_Fscore {res['T'][1]:.2f}" 
              )
        return

    # 初始化trainer
    trainer = ModelTrainer(device, **vars(args))

    # 加载预训练模型
    if args.initial_model != "":
        print("Model %s loaded from previous state!" % args.initial_model)
        trainer.load_parameters(args.initial_model)

    # 模型训练
    Fscores = []
    best_score = 0

    for epoch in range(1, args.epoch + 1):
        print("-" * 10, f'第{epoch}轮训练开始', "-" * 10)
        # 训练
        lr, loss = trainer.train_network(epoch, train_loader)
        # 测试
        if epoch % args.test_step == 0:
            trainer.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)
            res = trainer.eval_network(test_loader)

            fscore = res['M'][1]
            Fscores.append(fscore)
            if fscore > best_score:
                trainer.save_parameters(args.model_save_path + "/best_model.model")
                best_score = fscore
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                f"epoch {epoch}, Loss {loss:.2f}, Lr {lr:.6f}, " \
                f"M_Acc {res['M'][0]:.2f}, M_Fscore {res['M'][1]:.2f}, " \
                f"A_Acc {res['A'][0]:.2f}, A_Fscore {res['A'][1]:.2f}, " \
                f"T_Acc {res['T'][0]:.2f}, T_Fscore {res['T'][1]:.2f}, " \
                f"Best Fscore: {max(Fscores):.2f} [{(argmax(np.array(Fscores)) + 1) * args.test_step}epoch]"
                    )
            score_file.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                            f" —— {epoch} epoch, LR {lr:.6f}, LOSS {loss:.2f}, " \
                            f"M_Acc {res['M'][0]:.2f}, M_Fscore {res['M'][1]:.2f}, " \
                            f"A_Acc {res['A'][0]:.2f}, A_Fscore {res['A'][1]:.2f}, " \
                            f"T_Acc {res['T'][0]:.2f}, T_Fscore {res['T'][1]:.2f}, " \
                            f"Best Fscore: {max(Fscores):.2f} [{(argmax(np.array(Fscores)) + 1) * args.test_step}epoch]\n")
            score_file.flush()

    score_file.write('\n')
    score_file.close()


def parser_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='SpearkEmotionRecognition')

    # 训练参数
    parser.add_argument('--epoch', type=int, default=80, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of loader threads')
    parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
    parser.add_argument('--eval', type=bool, default=False, help='eval on-off')
    parser.add_argument('--initial_model', type=str, default='', help='Path of the initial_model file')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.98, help='Learning rate decay every [test_step] epochs')
    parser.add_argument("--seed", type=int, default=111, help='seed everything')
    parser.add_argument('--loss', type=str, default='CrossEntropy', help='CrossEntropy / NLL')
    parser.add_argument('--mult_task', type=str, default='MAT', help='MAT')

    # 数据 / 保存路径   
    parser.add_argument('--feature_path', type=str, default='/home/xwliang/workspaces/MyERC/UMERIC/features/', help='Path of the features file')
    parser.add_argument('--dataset', type=str, default='IEMOCAP', help='IEMOCAP / MELD')
    parser.add_argument('--save_path', type=str, default="exps/silstm",
                        help='Path to save the score.txt and models')

    # 模型参数
    parser.add_argument('--model', type=str, default='SILSTM', help='mdoel name')
    parser.add_argument('--n_classes', type=int, default=6, help='classes nums')
    parser.add_argument('--fusion_type', type=str, default='tfn', help='lmf / tfn / lfdnn / wf')

    # 选择显卡
    parser.add_argument("--gpu", type=int, default=3)

    args = parser.parse_args()
    args = init_args(args)
    
    return args


if __name__ == '__main__':
    # 解析参数
    args = parser_args()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Dataset:', args.dataset)
    print('Models:', args.mult_task)
    print('Cuda num:', args.gpu)
    print('Save path:', args.save_path)

    st = time.time()
    main(args)

    print(f'模型训练结束，共耗时：{round(time.time() - st, 2)}s')


