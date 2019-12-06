import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='StackGAN parameters')
parser.add_argument('--seed', default=8, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--epochs', default=600, type=int)
parser.add_argument('--save_interval', default=5, type=int)
parser.add_argument('--z_dim', default=100, type=int)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=True, help="use cuda or not")
parser.add_argument('--food_type', default='muffin')
parser.add_argument('--data_dir', default='./data')
parser.add_argument('--img_dir', default='./data/images')
parser.add_argument('--retrieval_model', default='models/cross_model.ckpt') # start from ../
parser.add_argument('--base_size', default=64, type=int)
parser.add_argument('--levels', default=2, type=int)
parser.add_argument('--phase', default='train', type=str, choices=['train', 'val', 'test'])
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--lr_g', default=2e-4, type=float)
parser.add_argument('--lr_d', default=2e-4, type=float)
parser.add_argument('--beta0', default=0.5, type=float)
parser.add_argument('--beta1', default=0.999, type=float)
parser.add_argument("--bi_condition", type=str2bool, nargs='?', const=True, default=True, help="use bi_condition or not")
parser.add_argument('--weight_uncond', default=1.0, type=float)
parser.add_argument('--weight_kl', default=0.02, type=float)
parser.add_argument('--weight_cycle_img', default=0.0, type=float)
parser.add_argument('--weight_cycle_txt', default=0.0, type=float)
parser.add_argument('--weight_tri_loss', default=10.0, type=float)
parser.add_argument('--labels', default='original', type=str, choices=['original', 'R-smooth', 'R-flip', 'R-flip-smooth'])
parser.add_argument("--input_noise", type=str2bool, nargs='?', const=True, default=False, help="add noise N(0,0.1) to input images in D. Sigma is decreased linearly between epochs 0-200.")

parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="in debug mode or not")
# for eval_StackGANv2.py
parser.add_argument('--splits', default=2, type=int)
args = parser.parse_args()