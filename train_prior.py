import logging
import argparse
import json
import os
import pandas as pd
from typing import List
import torch
from model.minGPT import GPT, GPTConfig, save_gpt_config, load_gpt_model
from prior.trainer import Trainer, TrainerConfig
from utils.utils import set_random_seed
from utils.dataset_PriorGPT import load_seqs_from_list_con, get_tensor_dataset, AASeqDictionary_con
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger.addHandler(logging.NullHandler())


def load_pretrain_model(prior_path, device='cuda'):
	logger.info("Loading pretrained models")
	model_def = Path(prior_path).with_suffix('.json')
	logger.info(f"Loading prior & agent to device {device}")
	try:
		prior = load_gpt_model(model_def, prior_path, device, copy_to_cpu=False)
		return prior
	except:
		raise Exception(f"Device '{device}' or model not available")


def train(training_set, validation_set, column_cdrs,column_cons,output_dir, n_epochs=10, lr=1e-3, batch_size=512,
		  n_layer=8, n_embd=512, n_head=8, max_len=14, device='cpu', num_workers=1, seed=42, model_path=None,con_num=3):

	logger.info(f'Running device:\t{device}')
	device = torch.device(device)
	set_random_seed(seed, device)
	logger.info(len(training_set))

	# load data
	train_seqs = load_seqs_from_list_con(training_set, column_cdrs,column_cons,max_len=max_len, rm_duplicates=True)
	valid_seqs = load_seqs_from_list_con(validation_set,column_cdrs,column_cons, max_len=max_len, rm_duplicates=True)
	logger.info(len(train_seqs))
	logger.info(len(valid_seqs))

	train_set = get_tensor_dataset(train_seqs)
	test_set = get_tensor_dataset(valid_seqs)

	sd = AASeqDictionary_con()
	n_characters = sd.get_char_num()
	block_size = max_len + 2   # add start & end

	# build network
	if model_path is not None:
		model = load_pretrain_model(model_path, device=device)
		mconf = GPTConfig(n_characters, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd,con_num=con_num)
		save_gpt_config(mconf, output_dir, 'gpt_model_config')  # save config for later use
	else:
		mconf = GPTConfig(n_characters, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd,con_num=con_num)
		model = GPT(mconf)
		save_gpt_config(mconf, output_dir, 'gpt_model_config')  # save config for later use

	# initialize a trainer instance and kick off training
	tconf = TrainerConfig(learning_rate=lr, lr_decay=True, warmup_tokens=0.1*len(train_set)*max_len,
						  final_tokens=n_epochs*len(train_set)*max_len, output_dir=output_dir)
	trainer = Trainer(model, tconf)
	trainer.fit(train_set, test_set,
				n_epochs=n_epochs, batch_size=batch_size, num_workers=num_workers, save_model=True)
	return trainer.model


def main(args):
	df_train = pd.read_csv(args.train_data)
	df_valid = pd.read_csv(args.valid_data)

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	with open(os.path.join(args.output_dir, 'commandline_args.json'), 'w') as f:
		json.dump(args.__dict__, f, indent=2)

	column1 = 'aa_seqs'
	columns = ['raw_FvNetCharge', 'raw_FvCSP', 'raw_HISum']

	logger.info(f"Training prior model started, the results are saved in {args.output_dir}")
	train(training_set=df_train, validation_set=df_valid,column_cdrs=column1,column_cons=columns,
		  output_dir=args.output_dir, n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
		  n_layer=args.n_layers, n_embd=args.n_embd, n_head=args.n_head,
		  device=args.device, max_len=args.max_len, seed=args.seed,model_path=args.model_path,con_num=args.con_num)

	logger.info(f'Training done, the trained model is in {args.output_dir}')


def parse_args():

	parser = argparse.ArgumentParser(description='Train prior GPT model on sequence',
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--train_data', '-t', type=str, help='Full path to sequence file containing training data',
					 default='data/OAS/oas_train.csv')
	parser.add_argument('--valid_data', '-v', type=str, help='Full path to sequence file containing validation data',
					 default='data/OAS/oas_test.csv')
	parser.add_argument('--output_dir', '-o', type=str, help='Output directory',default='results/42/prior/')

	optional = parser.add_argument_group('Optional')
	optional.add_argument('--n_epochs', default=10, type=int, help='Number of training epochs, default=10')
	optional.add_argument('--lr', default=1e-3, type=float, help='GPT learning rate, default=1e-3')
	optional.add_argument('--n_layers', default=8, type=int, help='Number of layers for training, default=8')
	optional.add_argument('--batch_size', default=1024, type=int, help='Size of batch for training, default=1024')
	optional.add_argument('--n_embd', default=256, type=int, help='Number of embeddings for GPT model, default=256')
	optional.add_argument('--n_head', default=8, type=int, help='Number of attention heads for GPT model, default=8')
	optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
	optional.add_argument('--max_len', default=14, type=int, help='Max length of a sequence, default=14')
	optional.add_argument('--model_path', default=None, type=str, help='Prior model path to fine-tune')
	optional.add_argument('--con_num', default=3, type=int, help='condition num')
	optional.add_argument('--seed', default=42, type=int, help='random seed')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	main(args)
