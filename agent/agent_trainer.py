import os
import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from utils.dataset_PriorGPT import AASeqDictionary_con,Experience
from utils.utils import unique,fraction_valid_seqs,set_random_seed
from utils.dataset_PriorGPT import rnn_start_token_vector
from model.minGPT import load_gpt_model, save_gpt_model
from agent.scoring_functions import ScoringFunctions
from agent.scoring.template import FVTemplate
import pickle
import time

class TrainerConfig:
    # optimization parameters
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    output_dir = '../'
    # print("load TrainerConfig......")

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def save_experience(experience, filename):
    with open(filename, 'wb') as f:
        pickle.dump(experience, f)

def load_experience(filename):
    with open(filename, 'rb') as f:
        experience = pickle.load(f)
    return experience

def save_random_state(seed, device, filename):
    random_state = {
        'seed': seed,
        'device': device,
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'torch_cuda_random_state': torch.cuda.get_rng_state_all() if device == 'cuda' else None
    }
    with open(filename, 'wb') as f:
        pickle.dump(random_state, f)

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

def modified_cosine_annealing_decay(optimizer, initial_lr, final_lr, step, total_steps, factor=0.5):
    """
    Modified Cosine annealing decay from initial_lr to final_lr, with the rapid drop-off point shifted.
    The `factor` parameter controls how much the decay is shifted.
    """
    cosine_decay = 0.5 * (1 + np.cos(np.pi * ((step / total_steps) ** factor)))
    step_lr = final_lr + (initial_lr - final_lr) * cosine_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = step_lr

def load_random_state(filename):
    with open(filename, 'rb') as f:
        random_state = pickle.load(f)
    np.random.set_state(random_state['numpy_random_state'])
    torch.manual_seed(random_state['seed'])
    if random_state['device'] == 'cuda':
        torch.cuda.manual_seed(random_state['seed'])
        torch.cuda.manual_seed_all(random_state['seed'])
        torch.set_rng_state(random_state['torch_random_state'])
        torch.cuda.set_rng_state_all(random_state['torch_cuda_random_state'])

def save_optimizer_state_as_pkl(optimizer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(optimizer.state_dict(), f)

def load_optimizer_state_from_pkl(optimizer, filename):
    with open(filename, 'rb') as f:
        state_dict = pickle.load(f)
    optimizer.load_state_dict(state_dict)

def calculate_is_success(row):
    her2 = row['raw_HER2']
    mhc2 = row['raw_MHC2']
    fv_net_charge = row['raw_FvNetCharge']
    fv_csp = row['raw_FvCSP']
    hisum = row['raw_HISum']

    if her2 > 0.7 and mhc2 > 2.51 and fv_net_charge > 0 and fv_net_charge < 6.2 and fv_csp > 6.61 and hisum >= 0 and hisum <= 4:
        return 'TRUE'
    else:
        return 'FALSE'
    
logger = logging.getLogger(__name__)
import pandas

class AgentTrainer:
    def __init__(self, prior_path, agent_path,exp_path, seed_state_path,opt_path,seed,save_dir, device='cuda', learning_rate=0.0001, batch_size=64, n_steps=3000,
                 sigma=60, max_seq_len=100, score_fns=None, weights=None, score_type='weight', df_min_score=0.4,
                 experience_replay=False, exp_max_size=50):
        logger.info("Initializing agent trainer...")
        self.prior_path = prior_path
        self.agent_path = agent_path
        self.exp_path=exp_path
        self.save_dir = save_dir
        self.seed_state_path= seed_state_path
        self.opt_path=opt_path
        self.seed=seed
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.sigma = sigma
        self.experience_replay = experience_replay
        if experience_replay:
            self.experience = Experience(max_size=exp_max_size)  # Init experience
        self.max_seq_len = max_seq_len
        self.sd = AASeqDictionary_con()

        set_random_seed(self.seed,self.device)
        self.prior_model, self.agent_model = self.load_pretrain_models()
        self.prior_model = self.prior_model.module if hasattr(self.prior_model, "module") else self.prior_model
        self.agent_model = self.agent_model.module if hasattr(self.agent_model, "module") else self.agent_model
        self.tconf = TrainerConfig(learning_rate=self.learning_rate, lr_decay=True)
        self.optimizer = self.agent_model.configure_optimizers(self.tconf)  # Use adamW with lr_decay
        if self.opt_path:
            load_optimizer_state_from_pkl(self.optimizer, self.opt_path)
        self.agent_model = torch.nn.DataParallel(self.agent_model).to(self.device)  # Enable using multiple GPUs
        self.prior_model = torch.nn.DataParallel(self.prior_model).to(self.device)
        self.final_df=pandas.DataFrame()

        if score_fns is None:
            score_fns, weights = ['HER2'], [1]

        herceptin = FVTemplate(
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS',
        'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK',
        'SRWGGDGFYAMDY',
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC',
        'WGQGTLVTVSS',
        'QDVNTA', 'QQHYTTPPT', 'SR', 'Y')

        self.scoring_function = ScoringFunctions(score_fns, weights=weights,template=herceptin)
        self.score_type = score_type #weight

        self.writer = SummaryWriter(self.save_dir)

    def load_pretrain_models(self):
        logger.info("Loading pretrained models")
        model_def = Path(self.prior_path).with_suffix('.json')
        logger.info(f"Loading prior & agent to device {self.device}")
        try:
            prior = load_gpt_model(model_def, self.prior_path, self.device, copy_to_cpu=False)
            agent = load_gpt_model(model_def, self.agent_path, self.device, copy_to_cpu=False)
            if self.exp_path:
                self.experience = load_experience(self.exp_path)
            if self.seed_state_path:
                load_random_state(self.seed_state_path)
            return prior, agent
        except:
            raise Exception(f"Device '{self.device}' or model not available")
        
    def nll_loss(self, inputs, targets):
        """
            Custom Negative Log Likelihood loss that returns loss per example, rather than for the entire batch.

            Args:
                inputs : (batch_size, num_classes) *Log probabilities of each class*
                targets: (batch_size) *Target class index*

            Outputs:
                loss : (batch_size) *Loss for each example*
        """
        target_expanded = torch.zeros(inputs.size()).to(self.device)
        target_expanded.scatter_(1, targets.contiguous().view(-1, 1).detach(), 1.0)  # One_hot encoding
        loss = torch.sum(target_expanded * inputs, 1)
        return loss

    def sample(self, model, num_samples: int ):
        """
            Sample molecules from agent and calculate likelihood
            Args:
                model: model to sample from
                num_samples: number of samples to produce for each step, i.e. batch_size

            Returns:
                sample_idxes: a list of SMILES indexes, with no beginning nor end symbols
                log_probs: log likelihood for SMILES generated
            """
        x = rnn_start_token_vector(num_samples, self.device)
        finished = torch.zeros(num_samples).byte().to(self.device)
        
        values = torch.LongTensor([24, 26, 28]).to(self.device)
        values = values.view(1, -1).expand(num_samples, -1)
        x = torch.cat([x, values], dim=1)
        sequences = x
        log_probs = torch.zeros(num_samples).to(self.device)
        for step in range(self.max_seq_len):
            logits, _ = model(x)
            prob = F.softmax(logits[:, -1, :], dim=-1)  # only for last time-step
            if step>=2 and step!=12:
                sampled_idx = Categorical(probs=prob).sample().squeeze()
            elif step==0:
                sampled_idx =  torch.full((num_samples,1), 18).to(self.device) # batch_size个S(18)
            elif step==1:
                sampled_idx =  torch.full((num_samples,1), 4).to(self.device) # batch_size个R(4)
            elif step==12:
                sampled_idx =  torch.full((num_samples,1), 21).to(self.device) # batch_size个R(4)
            sequences=torch.cat((sequences,sampled_idx.view(-1, 1)),1)
            x = sequences

            log_probs += self.nll_loss(prob.log(), sampled_idx)  # update log_probs

        return sequences.detach(), log_probs

    def likelihood(self, model, sample_idxes):
        """
        Retrieves the likelihood of a given sequence
            Args: x
                model: GPT model to calculate likelihood
                sample_idxes: A list of smiles of batch_size length
                device: Device used
            Outputs:
                log_probs : (batch_size) Log likelihood for each example
        """

        x = sample_idxes.to(self.device)
        num_samples, seq_length = x.size()
        log_probs = torch.zeros(num_samples).to(self.device)

        for step in range(4, seq_length):
            logits, _ = model(x[:, :step])
            log_prob = F.log_softmax(logits[:, -1, :], dim=-1).squeeze()
            log_probs += self.nll_loss(log_prob, x[:, step])

        return log_probs
    
    def replay_experience(self, loss, agent_likelihoods, prior_likelihoods, seqs, scores):
        if len(self.experience) > 4:  # Sample experiences and update loss
            exp_smiles, exp_scores, exp_prior_likelihoods = self.experience.sample(4)
            nums=len(exp_smiles)
            values = torch.LongTensor([1,24, 26, 28])
            values = values.view(1, -1).expand(nums, -1)
            exp_smiles = torch.cat([values, exp_smiles], dim=1)
            exp_agent_likelihoods = self.likelihood(self.agent_model, exp_smiles) 
            exp_augmented_likelihood = exp_prior_likelihoods + self.sigma * exp_scores
            exp_augmented_likelihood = torch.from_numpy(exp_augmented_likelihood).to(self.device)
            exp_loss = torch.pow((exp_augmented_likelihood - exp_agent_likelihoods), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihoods = torch.cat((agent_likelihoods, exp_agent_likelihoods), 0)
            
        prior_likelihoods = prior_likelihoods.data.cpu().numpy()
        new_experience = zip(seqs, scores, prior_likelihoods) 
        self.experience.add_experience(new_experience)  # Add new experience
        return loss, agent_likelihoods
    
    def save_step(self, step, scores_df, agent_likelihoods, prior_likelihoods, augmented_likelihoods):
        """
            Save step to a CSV file
        """
        scores_df['step'] = step * np.ones(len(scores_df))
        scores_df['agent_likelihood'] = agent_likelihoods.data.cpu().numpy()
        scores_df['prior_likelihood'] = prior_likelihoods.data.cpu().numpy()
        scores_df['augmented_likelihood'] = augmented_likelihoods.data.cpu().numpy()
        scores_df['is_success'] = scores_df.apply(calculate_is_success, axis=1)
        self.final_df=pandas.concat([self.final_df, scores_df], ignore_index=True)
        
    
    def train(self):
        for param in self.prior_model.parameters():  
            param.requires_grad = False

        logger.info("Starting training agent...")
        for step in range(self.n_steps):
            st=time.time()
            sample_idxes, agent_likelihoods = self.sample(self.agent_model, self.batch_size)  # Sample from agent
            uniq_ids = unique(sample_idxes)  # Remove duplicates
            uniq_token_seqs = sample_idxes[uniq_ids] 
            seqs,idxs=self.sd.matrix_to_seqs_final(uniq_token_seqs,uniq_ids) 
            agent_likelihoods = agent_likelihoods[idxs] 
            mid_sample_idxs=sample_idxes[idxs]
            prior_likelihoods = self.likelihood(self.prior_model, mid_sample_idxs)

            scores_df = self.scoring_function.scores(seqs, step, score_type=self.score_type)
            scores = scores_df[self.score_type].to_numpy() 

            augmented_likelihoods = prior_likelihoods + self.sigma * torch.from_numpy(scores).to(self.device)
            loss = torch.pow((augmented_likelihoods - agent_likelihoods), 2)

            if self.experience_replay:
                loss, agent_likelihood = self.replay_experience(loss, agent_likelihoods, prior_likelihoods, seqs, scores)

            loss = loss.mean()
            loss -= 5 * 1e3 * (1/agent_likelihoods).mean()  # Penalize small likelihoods, stimulate learning

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent_model.parameters(), self.tconf.grad_norm_clip)
            self.optimizer.step()

            logger.info(f"Step {step}, Valid %: {fraction_valid_seqs(seqs) * 100:4.1f}, "
                        + f"Max score: {max(scores):6.2f}, Mean score: {scores.mean():6.2f}, ")
            self.writer.add_scalar('Valid (%)', fraction_valid_seqs(seqs) * 100, step + 1)
            self.writer.add_scalar('Max score', max(scores), step + 1)
            self.writer.add_scalar('Mean score', scores.mean(), step + 1)

            for i in range(8):
                print(seqs[i])

            self.save_step(step, scores_df, agent_likelihoods, prior_likelihoods, augmented_likelihoods)

            if (step+1) %1000==0: 
                save_gpt_model(self.agent_model, f'{self.save_dir}/model/', f'Agent_mpo_{step+1}')
                save_experience(self.experience, f'{self.save_dir}/state/experience_data_{step+1}.pkl')
                save_random_state(self.seed, self.device, f'{self.save_dir}/state/random_state_{step+1}.pkl')
                save_optimizer_state_as_pkl(self.optimizer, f'{self.save_dir}/state/optimizer_state_{step+1}.pkl')
                self.final_df.to_csv(os.path.join(f'{self.save_dir}/model/', f"mpo_{step+1}_step_scores.csv"), index=False)

            ed=time.time()
            print(f'use time {ed-st}s')