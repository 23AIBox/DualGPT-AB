# Adapted from https://github.com/MolecularAI/Reinvent
import numpy as np
import pandas as pd
from agent.scoring.HER2Specificity import Specificity2HER2
from agent.scoring.ViscosityClearance import FvNetCharge, FvCSP, HISum
from agent.scoring.MHCAffinity import MHCIIAffinity
from agent.scoring.template import FVTemplate
from typing import List
import logging


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoringFunctions:
    def __init__(self, scoring_func_names=None, weights=None, template=None):
        """
            scoring_func_names: List of scoring function names, default=['HER2']
            weights: List of int weights for each scoring function, default=[1]
        """
        self.scoring_func_names = ['HER2'] if scoring_func_names is None else scoring_func_names
        self.weights = np.array([1] * len(self.scoring_func_names) if weights is None else weights)
        self.all_funcs = {'HER2': Specificity2HER2, 'MHC2': MHCIIAffinity,
                          'FvNetCharge': FvNetCharge, 'FvCSP': FvCSP, 'HISum': HISum}
        self.template = template

    def scores(self, aa_seqs: List, step: int, score_type='sum'):
        scores, raw_scores = [], []
        for fn_name in self.scoring_func_names:
            score, raw_score =self.all_funcs[fn_name](template=self.template)(aa_seqs)
            scores.append(score)
            raw_scores.append(raw_score) 
        scores = np.float32(scores).T
        raw_scores = np.float32(raw_scores).T

        if score_type == 'sum':
            final_scores = scores.sum(axis=1)
        elif score_type == 'product':
            final_scores = scores.prod(axis=1)
        elif score_type == 'weight':
            final_scores = (scores * self.weights / self.weights.sum()).sum(axis=1)
        else:
            raise Exception('Score type error!')

        np_step = np.ones(len(aa_seqs)) * step
        scores_df = pd.DataFrame({'step': np_step, 'aa_seqs': aa_seqs, score_type: final_scores})
        scores_df[self.scoring_func_names] = pd.DataFrame(scores, index=scores_df.index)
        raw_names = [f'raw_{name}' for name in self.scoring_func_names]
        scores_df[raw_names] = pd.DataFrame(raw_scores, index=scores_df.index)
        return scores_df

def unit_tests():
    herceptin = FVTemplate(
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS',
        'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK',
        'SRWGGDGFYAMDY',
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC',
        'WGQGTLVTVSS',
        'QDVNTA', 'QQHYTTPPT', 'SR', 'Y')
    sf = ScoringFunctions(template=herceptin, scoring_func_names=['HER2','MHC2','FvNetCharge', 'FvCSP','HISum'])
    aa_seqs = ['SRYGCCEEIAFVY']
    print(sf.scores(aa_seqs, step=1))

if __name__ == "__main__":
    unit_tests()
