import yaml
import typing as tp
import os

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
# import wandb  # sudo apt install libpython3.7-dev python3.7 -m pip install wandb
import numpy as np

from datasets.timit import TimitTrain, TimitEval
from model.model import MS_SincResNet, SincNet
from utils import NestedNamespace, compute_chunk_info
import matplotlib.pyplot as plt


def PlotAccuracy(Accuracy, verbose_every):
    x = [x*100 for x in range(0, len(Accuracy)
                              * verbose_every, verbose_every)]
    y = [y*100 for y in Accuracy]
    plt.plot(x, y, label='Accuracy', color='green')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy [%]')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.title('Accuracy')
    path = "currentAccuracy.png"
    plt.savefig(path)
    plt.show()
    plt.close()


def PlotLoss(loss, verbose_every):
    x = [x for x in range(0,
                          len(loss)*verbose_every, verbose_every)]
    y = [y*100 for y in loss]
    plt.plot(x, y, label='Loss', color='red')
    plt.xlabel('epoch')
    plt.ylabel('Loss [%]')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.title('Loss')
    path = "currentLoss.png"
    plt.savefig(path)
    plt.show()
    plt.close()


def compute_accuracy(logits: torch.Tensor, labels: tp.Union[torch.Tensor, int]) -> float:
    # 0-1
    return torch.mean((torch.argmax(logits, dim=1) == labels).float()).item()


def main(params: NestedNamespace):
    # PlotLoss([0.15, 0.3, 0.45], params.verbose_every)
    TestAcc = []
    LossPlot = []
    chunk_len, chunk_shift = compute_chunk_info(params)
    dataset_train = TimitTrain(params.data.timit.path, chunk_len=chunk_len)
    dataset_evaluation = TimitEval(
        params.data.timit.path, chunk_len, chunk_shift, 'test.scp')
    dataloader = DataLoader(dataset_train, batch_size=params.batch_size,
                            shuffle=True, drop_last=True)

    # sinc_net = MS_SincResNet()
    sinc_net = SincNet(
        chunk_len, params.data.timit.n_classes, params.model.type)
    sinc_net = sinc_net.to(params.device)
    optim = torch.optim.RMSprop(
        sinc_net.parameters(), lr=params.lr, alpha=0.95, eps=1e-8)
    prev_epoch = 0
    if params.model.pretrain is not None:
        checkpoint = torch.load(params.model.pretrain,
                                map_location=torch.device(params.device))
        sinc_net.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch'] + 1
    criterion = nn.CrossEntropyLoss()

    for i in range(prev_epoch, prev_epoch + params.n_epochs):
        accuracy, losses = [], []
        sinc_net.train()
        for j, batch in enumerate(dataloader):
            optim.zero_grad()
            chunks, labels = batch
            chunks, labels = chunks.to(params.device), labels.to(params.device)
            logits = sinc_net(chunks)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()

            if i % params.verbose_every == 0:
                losses.append(loss.item())
                accuracy.append(compute_accuracy(logits, labels))
        print(f"epoch {i} ")

        if i % params.verbose_every == 0:
            sinc_net.eval()
            with torch.no_grad():
                chunks_accuracy, losses_test = [], []
                wavs_accuracy = 0
                for chunks, label, n_chunks in dataset_evaluation:
                    chunks = chunks.to(params.device)
                    logits = sinc_net(chunks)
                    loss = criterion(logits, torch.Tensor(
                        [label] * n_chunks).long().to(params.device))
                    losses_test.append(loss.item())
                    chunks_accuracy.append(
                        compute_accuracy(logits, label))  # 0-1
                    wavs_accuracy += (torch.argmax(logits.sum(dim=0))  # 0 or 1
                                      == label).item()
                TestAcc.append(wavs_accuracy/len(dataset_evaluation))
                LossPlot.append(np.mean(losses_test))
                print(f'epoch {i}\ntrain accuracy {np.mean(accuracy*100)}%\ntrain loss {np.mean(losses)} \n'
                      f'test loss {LossPlot[-1]}%\n'
                      f'test wav accuracy {(TestAcc[-1])*100}%')
                if len(TestAcc) > 1:
                    if TestAcc[-1] > TestAcc[-2]:
                        torch.save(
                            {'model_state_dict': sinc_net.state_dict(
                            ), 'optimizer_state_dict': optim.state_dict(), 'epoch': i},
                            os.path.join(params.save_path, params.model.type+'.pt'))
                        print("saved model!! ")
    PlotAccuracy(TestAcc, params.verbose_every)
    PlotLoss(LossPlot, params.verbose_every)
    print(TestAcc)
    print(LossPlot)
    f = open("TestAccLoss.txt", "w")
    f.write(f"test Acc: \n {TestAcc}\n")
    f.write(f"test Loss: \n {LossPlot}")


if __name__ == "__main__":
    with open('configs/cfg.yaml') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)
        params = NestedNamespace(params)
    if params.model.type not in ['cnn', 'sinc']:
        raise ValueError("Only two models are supported, use cnn or sinc.")
    if params.use_wandb:
        id = wandb.util.generate_id()
        print("id", id)
        wandb.init(project='SincNet', id=id, resume="allow",
                   config={'model type': params.model.type})
    main(params)


# def save_checkpoint(G,D,epoch,trained):
#     print("=> Saving checkpoint")
#     save_dir_g = os.path.join(G.model['General']['save_dir'], G.model['name']+ '_G.pt')
#     save_dir_d = os.path.join(D.model['General']['save_dir'], D.model['name']+ '_D.pt')
#     torch.save({'G_Model': G,'epoch': epoch,'trained':trained}, save_dir_g)
#     torch.save({'D_model':D}, save_dir_d)


# def load_checkpoint(load_dir,name):
#     load_dir_g= os.path.join(load_dir, name+'_G.pt')
#     load_dir_d= os.path.join(load_dir, name+'_D.pt')
#     print("=> Loading checkpoint")
#     checkpoint_G = torch.load(load_dir_g)
#     checkpoint_D = torch.load(load_dir_d)
#     gen,disc=checkpoint_G['G_Model'],checkpoint_D['D_model']
#     epoch,trained = checkpoint_G['epoch'],  checkpoint_G['trained']
#     if trained:
#         print('load model from file! validate on new genrated images :)\n')
#     else:
#         print('continue to train from epoch {}, see log dir for history :)\n'.format(epoch))
#     return  gen,disc, epoch,trained
