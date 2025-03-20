import os
import torch
import torch.nn.functional as F
from utils.utils import set_random_seed
from model.minGPT import GPT
from utils.dataset_PriorGPT import AASeqDictionary_con,rnn_start_token_vector
from tqdm import tqdm
from utils.dataset_EnhancedGPT import AASeqDictionary_con_2

from agent.scoring_functions import ScoringFunctions
from agent.scoring.template import FVTemplate
import pandas as pd
import numpy as np



def calculate_is_success_1(row):
    her2 = row['raw_HER2']
    mhc2 = row['raw_MHC2']
    fv_net_charge = row['raw_FvNetCharge']
    fv_csp = row['raw_FvCSP']
    hisum = row['raw_HISum']

    if her2 > 0.7 and mhc2 > 2.51 and fv_net_charge < 6.2 and fv_csp > 6.61 and hisum >= 0 and hisum <= 4:
        return 'TRUE'
    else:
        return 'FALSE'

def calculate_is_success(row):
    fv_net_charge = row['raw_FvNetCharge']
    fv_csp = row['raw_FvCSP']
    hisum = row['raw_HISum']

    if fv_net_charge < 6.2 and fv_csp > 6.61 and hisum >= 0 and hisum <= 4:
        return 'TRUE'
    else:
        return 'FALSE'

def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())
    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).detach(), 1.0)  # One_hot encoding
    loss = torch.sum(target_expanded * inputs, 1)
    return loss


def _sample_batch_1(model: GPT, batch_size: int, device, max_len, temperature) -> torch.Tensor:
    x= rnn_start_token_vector(batch_size, device) 
    values = torch.LongTensor([24, 26, 28]).to(device)
    values = values.view(1, -1).expand(batch_size, -1)
    x = torch.cat([x, values], dim=1)
    indices = torch.zeros((batch_size, max_len), dtype=torch.long).to(device) 
    tmp_sequences=x
    log_probs = torch.zeros(batch_size).to(device)
    for char in range(max_len):
        # x[B,T]
        logits, _ = model(x) #[B,1,vocab_size]
        logits = logits[:, -1, :] / temperature #[B,vocab_size]
        probs = F.softmax(logits, dim=-1) #[B,vocab_size]
        log_prob = F.log_softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probs=probs)
        action = distribution.sample() #[B]
        
        indices[:, char] = action.squeeze()
        tmp_sequences=torch.cat((tmp_sequences,indices[:, char].view(-1, 1)),1)
        x=tmp_sequences.clone()
        log_probs-=NLLLoss(log_prob,action)

    return indices,log_probs

def _sample_batch_2(model: GPT, batch_size: int, device, max_len, temperature) -> torch.Tensor:
    x= rnn_start_token_vector(batch_size, device) 
    values = torch.LongTensor([24, 26, 28,30,32]).to(device)
    values = values.view(1, -1).expand(batch_size, -1)
    x = torch.cat([x, values], dim=1)
    indices = torch.zeros((batch_size, max_len), dtype=torch.long).to(device) 
    tmp_sequences=x
    log_probs = torch.zeros(batch_size).to(device)
    for char in range(max_len):
        # x[B,T]
        logits, _ = model(x) #[B,1,vocab_size]
        logits = logits[:, -1, :] / temperature #[B,vocab_size]
        probs = F.softmax(logits, dim=-1) #[B,vocab_size]
        log_prob = F.log_softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probs=probs)
        action = distribution.sample() #[B]
        
        indices[:, char] = action.squeeze()
        tmp_sequences=torch.cat((tmp_sequences,indices[:, char].view(-1, 1)),1)
        x=tmp_sequences.clone()
        log_probs-=NLLLoss(log_prob,action)

    return indices,log_probs

def sample(voc:AASeqDictionary_con, model: GPT, num_to_sample=10000, device='cpu', batch_size=64, max_len=100, temperature=1.0, seed=42,con_num=3,out_path=None):
    model.eval()
    set_random_seed(seed, device)

    # Round up division to get the number of batches that are necessary:
    number_batches = (num_to_sample + batch_size - 1) // batch_size
    remaining_samples = num_to_sample

    indices = torch.LongTensor(num_to_sample, max_len).to(device)

    final_df = pd.DataFrame()

    herceptin = FVTemplate(
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS',
        'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK',
        'SRWGGDGFYAMDY',
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC',
        'WGQGTLVTVSS',
        'QDVNTA', 'QQHYTTPPT', 'SR', 'Y')
    weight = []
    weight.append(3.0 / 7)
    weight.append(1.0 / 7)
    weight.append(1.0 / 7)
    weight.append(1.0 / 7)
    weight.append(1.0 / 7)
    weight = np.array(weight)
    scoring_function = ScoringFunctions(template=herceptin,
                                        scoring_func_names=['HER2', 'MHC2', 'FvNetCharge', 'FvCSP', 'HISum'],
                                        weights=weight)

    model.eval()
    s=set()
    with torch.no_grad():
        batch_start = 0
        i=0
        while True:
            if con_num==3:
                indices,nll_loss = _sample_batch_1(model, batch_size, device, max_len, temperature)
                ori_idxs=torch.arange(batch_size).to(device)
            
                cdrs,idxs = voc.matrix_to_seqs(indices)
            elif con_num==5:
                indices,nll_loss = _sample_batch_2(model, batch_size, device, max_len, temperature)
                ori_idxs=torch.arange(batch_size).to(device)
            
                cdrs,idxs = voc.matrix_to_seqs_final(indices,ori_idxs.cpu().numpy())        
            
            new_list=[]
            for x in cdrs:
                if x not in s:
                    s.add(x)
                    new_list.append(x)
            score_df = scoring_function.scores(new_list, i, 'sum')
            score_df['is_success'] = score_df.apply(calculate_is_success_1, axis=1)
            final_df = pd.concat([final_df, score_df], ignore_index=True)
            remaining_samples = remaining_samples-len(new_list)
            if remaining_samples<=0:
                break
            i+=1
    
    num_to_discard = len(final_df) - num_to_sample
    discarded_df = final_df.sample(n=num_to_discard, random_state=10001)
    remaining_df = final_df.drop(discarded_df.index)
    remaining_df.to_csv(out_path, index=False)

from model.minGPT import load_gpt_model
from pathlib import Path

def load_pretrain_model(prior_path, device='cuda'):
    model_def = Path(prior_path).with_suffix('.json')
    try:
        prior = load_gpt_model(model_def, prior_path, device, copy_to_cpu=False)
        return prior
    except:
        raise Exception(f"Device '{device}' or model not available")


