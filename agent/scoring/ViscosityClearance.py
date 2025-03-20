import logging
import numpy as np
import pandas as pd
from agent.scoring.transformation import TransformFunction
from agent.scoring.template import FVTemplate

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_valid_seqs(aa_seqs: list):
    valid_idxes = []
    valid_seqs = []
    for idx, seq in enumerate(aa_seqs):
        if 'X' in seq:  # Ignore seq with unknown AAs
            continue
        valid_seqs.append(seq)
        valid_idxes.append(idx)
    return valid_seqs, valid_idxes

class SeqNetCharge:
    """
        Calculate net charge based on pKas and Henderson-Hasselbalch equation
    """
    def __init__(self, pH=5.5):
        self.pH = pH
        # The pKa values taken from: https://www.peptideweb.com/images/pdf/pKa-and-pI-values-of-amino-acids.pdf
        self.pKas = pd.DataFrame(
            [['A', 2.34, 9.69, None], ['C', 1.71, 10.78, 8.33], ['D', 2.09, 9.82, 3.86], ['E', 2.19, 9.67, 4.25],
             ['F', 1.83, 9.13, None], ['G', 2.34, 9.6, None], ['H', 1.82, 9.17, 6.04], ['I', 2.36, 9.68, None],
             ['K', 2.18, 8.95, 10.79], ['L', 2.36, 9.6, None], ['M', 2.28, 9.21, None], ['N', 2.02, 8.84, None],
             ['P', 1.99, 10.6, None], ['Q', 2.17, 9.13, None], ['R', 2.17, 9.04, 12.48], ['S', 2.21, 9.15, None],
             ['T', 2.63, 9.1, None], ['V', 2.32, 9.62, None], ['W', 2.38, 9.39, None], ['Y', 2.2, 9.11, 10.07]],
            columns=['AA', 'pKa-Ca-COOH', 'pKa-Ca-NH3+', 'pKa-side-chain'])
        self.aa_net_charge = self.get_AA_net_charge()

    def get_AA_net_charge(self):  # Net charge for AA based on Henderson-Hasselbalch equation
        DoD = 1 / (10 ** (self.pKas[['pKa-Ca-COOH', 'pKa-Ca-NH3+', 'pKa-side-chain']].values - self.pH) + 1)
        DoD[:, 0] = -1 * DoD[:, 0]  # COO-, negative charge
        DoD[:, 1] = 1 - DoD[:, 1]  # -NH3+, positive charge
        DoD[np.array([2, 3, 4, 20])-1, 2] = -1 * DoD[np.array([2, 3, 4, 20])-1, 2]  # COO-, negative charge
        DoD[np.array([7, 9, 15])-1, 2] = 1 - DoD[np.array([7, 9, 15])-1, 2]  # -NH3+, positive charge
        aa_net_charge_dict = dict(zip(self.pKas['AA'], np.nansum(DoD, axis=1)))
        return aa_net_charge_dict

    def __call__(self, seq: str):
        return np.sum(self.aa_net_charge[aa] for aa in seq)


class FvNetCharge:
    """
        Increasing antibody variable fragment (Fv) net charge is associated with decreased viscosity;
        the optimal Fv net charge for clearance is in [0, 6.2]. Sharma et al. 2015, PNAS.
        Input: list of  cdrh3
    """
    def __init__(self, template: FVTemplate, pH=5.5):
        self.seq_net_charge = SeqNetCharge(pH=pH)
        self.trans_fn = TransformFunction('dsigmoid', -0.2, 6.4, params={'k1': 1.5, 'k2': 1.5})
        self.template = template

    def __call__(self, aa_seqs: list):
        """
            aa_seqs: list of cdrh3
        """
        scores = np.zeros(len(aa_seqs))
        valid_seqs, valid_idxes = get_valid_seqs(aa_seqs)
        scores[valid_idxes] = np.array([self.seq_net_charge(self.template.fv_h_cdr3_prefix + seq
                                                            + self.template.fv_h_cdr3_suffix)
                                        + self.seq_net_charge(self.template.fv_l) for seq in valid_seqs])
        tf_scores = self.trans_fn(scores)
        return tf_scores, scores


class FvCSP:
    """
        Increasing antibody variable fragment charge symmetry parameter (FvCSP) is associated with decreased viscosity.
        Reference: Sharma et al. 2015, PNAS.
        Input: List of cdrh3.
    """

    def __init__(self, template: FVTemplate, pH=5.5):
        self.seq_net_charge = SeqNetCharge(pH=pH)
        self.trans_fn = TransformFunction('sigmoid', 0, 6, params={'k': 2.}) 
                                                    
        self.template = template
        # print(self.seq_net_charge(self.template.fv_l))

    def __call__(self, aa_seqs: list):
        """
            aa_seqs: list of cdrh3
        """
        scores = np.zeros(len(aa_seqs))
        valid_seqs, valid_idxes = get_valid_seqs(aa_seqs) 
        scores[valid_idxes] = np.array([self.seq_net_charge(self.template.fv_h_cdr3_prefix + seq
                                                            + self.template.fv_h_cdr3_suffix)
                                        * self.seq_net_charge(self.template.fv_l) for seq in valid_seqs])
        tf_scores = self.trans_fn(scores)
        return tf_scores, scores


class HISum:
    """
        Hydrophobicity index.
        The optimal HI sum of CDRL1, CDRL3 and CDRH3 for clearance is less than 4. Sharma et al. 2015, PNAS.
        Input: List of cdrh3. 
    """
    def __init__(self, template: FVTemplate):
        # AA classification based on Mason et al. (2021). Nature Biomedical Engineering, 5(6), 600-612. 
        self.phobic_aas = ['A', 'C', 'F', 'I', 'L', 'P', 'V', 'W', 'Y']
        self.philic_aas = ['D', 'E', 'G', 'H', 'K', 'M', 'N', 'Q', 'R', 'S', 'T']
        # Params taken from Eisenberg et al. 1984，Analysis of membrane and surface protein sequences with the hydrophobic moment plot
        self.philic_params = {'A': 0.25, 'C': 0.04, 'D': -0.72, 'E': -0.62, 'F': 0.61, 'G': 0.16, 'H': -0.4, 'I': 0.73,
                              'K': -1.1, 'L': 0.53, 'M': 0.26, 'N': -0.64, 'P': -0.07, 'Q': -0.69, 'R': -1.76,
                              'S': -0.26, 'T': -0.18, 'V': 0.54, 'W': 0.37, 'Y': 0.02}
        self.trans_fn = TransformFunction('dsigmoid', -1, 4.2, params={'k1': 1., 'k2': 1.})
        self.template = template

    def cal_HI(self, aa_seq):
        phobic_sum = sum(self.philic_params[aa] for aa in aa_seq if aa in self.phobic_aas)
        philic_sum = sum(self.philic_params[aa] for aa in aa_seq if aa in self.philic_aas)

        HI_score = -phobic_sum/philic_sum if philic_sum != 0.0 else np.nan
        return HI_score

    def __call__(self, aa_seqs: list):
        """
            combined cdrh3 with cdrl1 and cdrl3 from templated to calculate HI sum
            aa_seqs: List of cdrh3
        """
        scores = np.array([self.cal_HI(self.template.fv_l_cdr1) + self.cal_HI(self.template.fv_l_cdr3)
                           + self.cal_HI(seq) for seq in aa_seqs])
        tf_scores = self.trans_fn(scores)
        return tf_scores, scores


def unittest():
    aa_seq1 = 'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS'
    seq_net_charge = SeqNetCharge()
    print(seq_net_charge(aa_seq1))
    aa_seq2 = 'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK'
    print(seq_net_charge(aa_seq2))

    herceptin = FVTemplate(
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS',
        'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK',
        'SRWGGDGFYAMDY',
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC',
        'WGQGTLVTVSS',
        'QDVNTA', 'QQHYTTPPT', 'SR', 'Y')

    aa_seqs = ['SRWGGDGFYAMDY']
    fv_net_charge = FvNetCharge(template=herceptin)
    print(fv_net_charge(aa_seqs=aa_seqs))
    fv_csp = FvCSP(template=herceptin)
    print(fv_csp(aa_seqs=aa_seqs))

    aa_seqs = ['SRWGGDGFYAMDY', 'SRWGGDGFYAMDYW', 'SRAGVTGQYAYYY', 'SRRGTSGGYAFDY']
    hi_sum = HISum(template=herceptin)
    print(hi_sum(aa_seqs=aa_seqs))  


if __name__ == '__main__':
    unittest()
