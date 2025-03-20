# https://www.nature.com/articles/s41551-021-00699-9
# https://github.com/dahjan/DMS_opt

from agent.scoring.HER2.utils import create_cnn, one_hot_encoder
from agent.scoring.template import FVTemplate
from Bio.Alphabet import IUPAC
import numpy as np
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Specificity2HER2:
    def __init__(self, template: FVTemplate=None, weight_path='./HER2/CNN_HER2'):
        # initial the model and the weight
        params = [['CONV', 400, 5, 1], ['DROP', 0.2], ['POOL', 2, 1], ['FLAT'], ['DENSE', 300]]
        self.model = create_cnn(params, (10, 20), 'relu', None)
        self.model.load_weights(weight_path)
        self.template = template  # preserved parameter for future extension

    def pred_prob(self, aa_seqs):
        aa_seqs_enc = np.asarray([one_hot_encoder(s=seq, alphabet=IUPAC.protein) for seq in aa_seqs]) 
        aa_seqs_enc = np.transpose(aa_seqs_enc, (0, 2, 1))
        scores = self.model.predict(x=aa_seqs_enc)
        return list(scores.reshape(-1))

    def get_valid_seqs(self, aa_seqs: list, seq_len_flt=13):
        valid_idxes = []
        valid_seqs = []
        for idx, seq in enumerate(aa_seqs):
            if len(seq) != seq_len_flt or 'X' in seq or 'B' in seq:  # Ignore seq with unknown AAs, and only score for seq with 13 AAs
                continue
            if not (seq[:len(self.template.fv_h_cdr3_gen_prefix)] == self.template.fv_h_cdr3_gen_prefix and
                    seq[-len(self.template.fv_h_cdr3_gen_suffix):] == self.template.fv_h_cdr3_gen_suffix):  # seq pattern to include
                continue
            # Only take 10 AAs in specificity calculate, e.g. only 'WGGDGFYAMD' within 'SRWGGDGFYAMDY' is used
            valid_seqs.append(seq[len(self.template.fv_h_cdr3_gen_prefix):-len(self.template.fv_h_cdr3_gen_suffix)])
            valid_idxes.append(idx)
        return valid_seqs, valid_idxes

    def __call__(self, aa_seqs: list):
        scores = np.zeros(len(aa_seqs))
        valid_seqs, valid_idxes = self.get_valid_seqs(aa_seqs)
        if len(valid_seqs) > 0:
            scores[np.array(valid_idxes)] = self.pred_prob(valid_seqs)
        return scores, scores  # tf_scores & raw_scores


def unittest():
    herceptin = FVTemplate(
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS',
        'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK',
        'SRWGGDGFYAMDY',
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC',
        'WGQGTLVTVSS',
        'QDVNTA', 'QQHYTTPPT', 'SR', 'Y')

    affinity2HER2 = Specificity2HER2(template=herceptin, weight_path='/home/dnxie/ab-gen/agent/scoring/HER2/CNN_HER2')
    AASeqs = ['SRASGAVTDFVYY', 'SRASGAVTDFVAY', 'SRASGAVTDFVAAY', 'SRASGAVTDFVAA']
    print(affinity2HER2(aa_seqs=AASeqs))


if __name__ == '__main__':
    unittest()
