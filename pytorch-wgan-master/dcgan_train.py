from utils.config import parse_args
from utils.data_loader import get_data_loader
import torch
import torchvision
import torchvision.transforms as transforms
from dataloader import ImageLoader 
from models.dcgan import DCGAN_MODEL
from args import get_parser

parser = get_parser()
opt = parser.parse_args()

if not(torch.cuda.device_count()):
    device = torch.device(*('cpu',0))
else:
    torch.cuda.manual_seed(opt.seed)
    device = torch.device(*('cuda',0))

def main():
    model = None

    model = DCGAN_MODEL(opt,device)
    print("using DCGAN model")




    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    # cudnn.benchmark = True

    # preparing the training laoder
    train_loader = torch.utils.data.DataLoader(
        ImageLoader(opt.img_path,
            transforms.Compose([
            transforms.Scale(128), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(128), # we get only the center of that rescaled
            transforms.RandomCrop(128), # random crop within the center crop 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),data_path=opt.data_path,partition='train'),
        batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers, pin_memory=True)
    print('Training loader prepared.')


    # preparing validation loader 
    val_loader = torch.utils.data.DataLoader(
        ImageLoader(opt.img_path,
            transforms.Compose([
            transforms.Scale(128), # rescale the image keeping the original aspect ratio
            transforms.CenterCrop(128), # we get only the center of that rescaled
            transforms.ToTensor(),
            normalize,
        ]),data_path=opt.data_path,partition='val'),
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)
    print('Validation loader prepared.')

    # Start model training
    if opt.is_train == 'True':
        model.train(train_loader)
    else:
        print("Done!")


if __name__ == '__main__':
    main()