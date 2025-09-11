import os
import argparse
import logging
from pathlib import Path
import pandas as pd
from model.minGPT import load_gpt_model
from model.sampler import sample
import math
import os
from utils.dataset_PriorGPT import AASeqDictionary_con
from utils.dataset_EnhancedGPT import AASeqDictionary_con_2

def main(args):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'device:\t{args.device}')

    gpt_path = args.model_path
    out_path = args.out_file

    model_def = Path(gpt_path).with_suffix('.json')
    model = load_gpt_model(model_def, gpt_path, args.device, copy_to_cpu=True)
    logger.info(f'Generate samples...')
    num_to_sample = args.num_to_sample
    if model.con_num==5:
        voc=AASeqDictionary_con_2()
    elif model.con_num==3:
        voc=AASeqDictionary_con()
    sample_seqs = sample(voc,model, num_to_sample=num_to_sample, device=args.device, batch_size=args.batch_size,
                         max_len=args.max_len, temperature=args.temperature,seed=args.seed,con_num=model.con_num,out_path=args.out_file)
    uniq_seqs = list(set(sample_seqs))

    logger.info(f"Totally {len(uniq_seqs)} unique sequences!")
    logger.info(f'Generation finished!')

def get_args():
    parser = argparse.ArgumentParser(description='Generate CDRH3s from a GPT model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, help='Full path to GPT model',default='results/42/enhance/gpt_model_final.pt')
    parser.add_argument('--out_file', type=str, help='Output file path',default='results/42/enhance/DualGPT-AB_10k.csv')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--num_to_sample', default=10000, type=int, help='Molecules to sample, default=10000')
    optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
    optional.add_argument('--batch_size', default=64, type=int, help='Batch_size during sampling, default=64')
    optional.add_argument('--max_len', default=13, type=int, help='Maximum seqs length, default=13')
    optional.add_argument('--temperature', default=1.0, type=float, help='Temperature during sampling, default=1.0')
    optional.add_argument('--seed', default=42, type=int, help='random seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
