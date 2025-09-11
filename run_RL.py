import os
import argparse
import logging
import json
from agent.agent_trainer import AgentTrainer
import time


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)



def main(args):
    logger.info(f'device:\t{args.device}')
    logger.info('Training gpt agent started!')
    # output_dir = args.output_dir + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'commandline_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    if args.task == 'mpo':
        score_fns = ['HER2', 'MHC2','FvNetCharge', 'FvCSP', 'HISum'] if args.mpo else ['HER2']
        weights = [3.0/7, 1.0/7, 1.0/7, 1.0/7, 1.0/7] if args.mpo else [1]
    elif args.task == 'her2':
        score_fns = ['HER2'] 
        weights = [1]
    else:
        raise Exception("Task type not in ['mpo', 'her2']")
    
    trainer = AgentTrainer(prior_path=args.prior, agent_path=args.agent, exp_path=args.exp_path,seed_state_path=args.seed_state,opt_path=args.opt_path,seed=args.seed,save_dir=output_dir, device=args.device,
                           learning_rate=args.lr, batch_size=args.batch_size, n_steps=args.n_steps, sigma=args.sigma,
                           experience_replay=args.er, max_seq_len=args.max_len, score_fns=score_fns,weights=weights)
    trainer.train()
    logger.info(f"Training agent finished! Results saved to folder {output_dir}")
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', '-p', type=str, help='Path to prior checkpoint (.ckpt)',default='results/42/prior/gpt_model_final.pt')
    parser.add_argument('--agent', '-a', type=str, help='Path to agent checkpoint, likely prior (.ckpt)',default='results/42/prior/gpt_model_final.pt')
    parser.add_argument('--output_dir', '-o', type=str, help='Output directory',default='results/42/rl')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--task', '-t', type=str, default='mpo', help='Task to run: her2, mpo')
    optional.add_argument('--batch_size', type=int, default=64, help='Batch size (default is 64)')
    optional.add_argument('--n_steps', type=int, default=2000, help='Number of training steps (default is 3000)')
    optional.add_argument('--sigma', type=int, default=60, help='Sigma to update likelihood (default is 60)')
    optional.add_argument('--device', default='cuda', type=str, help='Use cuda or cpu, default=cuda')
    optional.add_argument('--lr', default=1e-4, type=float, help='Learning rate, default=1e-4')
    optional.add_argument('--er', action="store_true", help='Experience replay or not, default False',default=True)
    optional.add_argument('--mpo', action="store_true", help='Multiple properties or not, default False',default=True)
    optional.add_argument('--seed', default=42, type=int, help='Random Seed, default=42')
    optional.add_argument('--max_len', default=13, type=int, help='Maximum sequence length, default=13')
    optional.add_argument('--exp_path', default=None, type=str, help='experience file path')
    optional.add_argument('--seed_state', default=None, type=str, help='seed_state file path')
    optional.add_argument('--opt_path', default=None, type=str, help='optimizer file path')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)