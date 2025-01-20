import torch

from pathlib import Path
from argparse import ArgumentParser
from torch.optim import AdamW
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from tqdm import tqdm

from src.model import ResModel


def epoch(ep, model, loader, lf, opt=None, bt=2, train=True):
    runloss = 0
    n = len(loader)
    if train:
        model.train()
    else:
        model.eval()
    for batch in loader:
        inputs, _ = batch
        if train:
            opt.zero_grad()
        out = model(inputs, bt)
        loss = lf(out, inputs)
        runloss += loss.item()
        if train:
            loss.backward()
            opt.step()
    return runloss / n


def train(args):
    t1 = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize([0, 0, 0], [1, 1, 1])
    ])
    t2 = v2.Compose([
        v2.RandomHorizontalFlip(0.2),
        v2.RandomVerticalFlip(0.2),
        # v2.RandomPerspective(p=0.2),
        v2.RandomRotation([-30, 30]),
        # v2.RandomPosterize(6, p=0.2),
        v2.ColorJitter(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize([0, 0, 0], [1, 1, 1])
    ])
    data = ImageFolder(
        args.traindata,
        t1 if args.notrans else t2
    )
    loader = DataLoader(
        dataset=data,
        batch_size=args.bs,
        shuffle=True
    )

    testdata = ImageFolder(
        args.testdata,
        t1
    )
    testloader = DataLoader(
        dataset=testdata,
        batch_size=args.bs,
        shuffle=True
    )

    exp_path = Path('./exp')
    exp_path /= args.exp
    model_path = exp_path / 'models'
    model_path.mkdir(exist_ok=True, parents=True)
    eval_path = exp_path / 'eval'
    eval_path.mkdir(exist_ok=True, parents=True)

    model = ResModel()
    model.train()
    loss_fn = MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    aux_loss_fn = None

    with open(eval_path / 'log.csv', 'w') as logfile:
        for ep in tqdm(range(args.ep)):
            trloss = epoch(
                ep,
                model,
                loader,
                loss_fn,
                optimizer,
                bt=args.bt
            )
            print('Loss:', trloss)
            print('Loss', ep, trloss, sep=',', file=logfile)

            if ep % args.save == 0:
                teloss = epoch(
                    ep,
                    model,
                    testloader,
                    loss_fn,
                    train=False
                )
                print('ValLoss:', teloss)
                print('Val', ep, teloss, sep=',', file=logfile)
                torch.save(model.enc.state_dict(), model_path / f'enc{ep}.pt')
                torch.save(model.dec.state_dict(), model_path / f'dec{ep}.pt')
        teloss = epoch(
            ep,
            model,
            loader,
            loss_fn,
            train=False
        )
        print('ValLoss:', teloss)
        print('Val', ep, teloss, sep=',', file=logfile)
        torch.save(model.enc.state_dict(), model_path / f'enc{ep}.pt')
        torch.save(model.dec.state_dict(), model_path / f'dec{ep}.pt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp', default='res0')
    parser.add_argument('--traindata', default='./data/train')
    parser.add_argument('--testdata', default='./data/test')
    parser.add_argument('--bs', type=int, default=48)
    parser.add_argument('--ep', type=int, default=200)
    parser.add_argument('--bt', type=int, default=2)
    parser.add_argument('--b', type=int, default=2)
    parser.add_argument('--save', type=int, default=50)
    parser.add_argument('--notrans', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    print(type(args.notrans), args.notrans)

    train(args)
