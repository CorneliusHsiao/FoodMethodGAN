import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='retrieval model parameters')
parser.add_argument('--seed', default=8, type=int)
parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=True, help="use cuda or not")
parser.add_argument('--hid_dim', default=1024, type=int)
parser.add_argument('--z_dim', default=1024, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--workers', default=16, type=int)
parser.add_argument('--grad_clip', type=int, default=5)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--margin', default=0.3, type=float)
parser.add_argument('--upmc_model', default='./pretrain_upmc/models/upmc_resnet50.ckpt')
parser.add_argument('--data_dir', default='./data')
parser.add_argument('--img_dir', default='./data/images')
parser.add_argument('--retrieved_type', default='recipe', choices=['recipe', 'image'])
parser.add_argument('--retrieved_range', default=1000, type=int)
parser.add_argument('--val_freq', default=1, type=int)
parser.add_argument('--save_freq', default=2, type=int)
parser.add_argument('--resume', default='')

# only for train_retrieval_model.py
parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="in debug mode or not")

# only for eval_ingr_retrieval.py
parser.add_argument('--save_dir', default='./experiments')
parser.add_argument('--generation_model', default='generative_model/models/salad.ckpt')
parser.add_argument('--food_type', default='salad')
parser.add_argument('--hot_ingr', default='tomato')
args = parser.parse_args()

