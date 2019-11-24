import argparse

def get_parser():

	parser = argparse.ArgumentParser()
	# data
	parser.add_argument('--img_path', default='../data/img_data/')
	parser.add_argument('--data_path', default='../data/')
	parser.add_argument('--workers', default=16, type=int)
	parser.add_argument('--seed', default=1234, type=int)
	# model
	parser.add_argument('--snapshots', default='./snapshots/',type=str)
	parser.add_argument('--save_image', default='./images/')

	parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
	parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
	parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
	parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
	parser.add_argument("--embeding_dim", type=int, default=1024, help="dimensionality of the latent space")
	parser.add_argument("--latent_dim", type=int, default=1024, help="dimensionality of the latent space")
	parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
	parser.add_argument("--channels", type=int, default=3, help="number of image channels")
	parser.add_argument("--sample_interval", type=int, default=5, help="interval between image sampling")

	# save
	parser.add_argument("--save_img", type=str, default="./images_generated/")
	parser.add_argument("--save_model", type=str, default="./snapshots/")
	return parser