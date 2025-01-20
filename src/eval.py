from argparse import ArgumentParser
from model import ResModel


def load_torch(exp_name, ep):
    model = ResModel()
    pth = f'./exp/{exp_name}/models'
    model.enc.load_state_dict(f'{pth}/enc{ep}.pt')
    model.dec.load_state_dict(f'{pth}/dec{ep}.pt')
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--ep', required=True, type=int)
    args = parser.parse_args()

    eval(args)
