# https://academic.oup.com/nar/article/48/W1/W449/5837056#205007555
import errno
import logging
import os
import shutil

import numpy as np
import pandas as pd
import tempfile
from agent.scoring.template import FVTemplate
from agent.scoring.transformation import TransformFunction
from multiprocessing import Pool

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MHCIIAffinity:
    def __init__(self, template: FVTemplate, path='/home/dnxie/project/netMHCIIpan-4.1/netMHCIIpan.sh', num_proc=16):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.hla = 'DRB1_0101,DRB1_0301,DRB1_0401,DRB1_0405,DRB1_0701,DRB1_0802,DRB1_0901,DRB1_1101,DRB1_1201,' \
                   'DRB1_1302,DRB1_1501,DRB3_0101,DRB3_0201,DRB4_0101,DRB5_0101,HLA-DQA10501-DQB10201,' \
                   'HLA-DQA10501-DQB10301,HLA-DQA10301-DQB10302,HLA-DQA10401-DQB10402,HLA-DQA10101-DQB10501,' \
                   'HLA-DQA10102-DQB10602,HLA-DPA10201-DPB10101,HLA-DPA10103-DPB10201,HLA-DPA10103-DPB10401,' \
                   'HLA-DPA10104-DPB10401,HLA-DPA10105-DPB10401,HLA-DPA10106-DPB10401,HLA-DPA10107-DPB10401,' \
                   'HLA-DPA10108-DPB10401,HLA-DPA10109-DPB10401,HLA-DPA10110-DPB10401,HLA-DPA10301-DPB10402,' \
                   'HLA-DPA10201-DPB10501,HLA-DPA10201-DPB11401'  # HLAs to bind against
        self.template = template
        self.trans_fn = TransformFunction('sigmoid', 2, 10, params={'k': 1.5})
        self.num_proc = num_proc

    @staticmethod
    def gen_temp_file(suffix=None, prefix=None, dir=None, text=True) -> str:
        filehandler, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
        os.close(filehandler)
        return path

    def get_min_pct_rank(self, aa_seq: str, tmp_dir: str):
        temp_file_path = self.gen_temp_file(prefix='mhc2-', suffix='.fa')
        path1='/home/dnxie'
        path2=temp_file_path
        path2=path2[1:]
        temp_file_path=os.path.join(path1, path2)
        with open(temp_file_path, 'w') as w:
            w.write(f'>temp_name\n{aa_seq}\n')

        command = f"{self.path} -f {temp_file_path} -a {self.hla} -xls -xlsfile {temp_file_path} 2>&1 > /dev/null " \
                  f"&& sed -i \'1d\' {temp_file_path}"
        # logger.info(command)
        os.system(command)
        df = pd.read_csv(f"{temp_file_path}", sep='\t')
        filter_col = [col for col in df if col.startswith('Rank')]
        df_filtered = df[filter_col]
        score = df_filtered.min().min()
        os.system(f'rm {temp_file_path}')
        return score

    def get_min_pct_ranks_parallel(self, padded_seqs: list):
        tmp_dir = tempfile.mkdtemp()
        pool = Pool(processes=self.num_proc)
        sub_procs = []
        for seq in padded_seqs:
            proc = pool.apply_async(self.get_min_pct_rank, (seq, tmp_dir))
            sub_procs.append(proc)
        pool.close()
        pool.join()
        scores = np.array([proc.get() for proc in sub_procs])

        try:
            shutil.rmtree(tmp_dir)
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise Exception("Error while remove temp dir!")

        return scores

    def __call__(self, aa_seqs: list, padding=10) -> np.array:
        padded_seqs = [self.template.fv_h_cdr3_prefix[-padding:] + seq + self.template.fv_h_cdr3_suffix[:padding]
                       for seq in aa_seqs] 
        scores = self.get_min_pct_ranks_parallel(padded_seqs)
        tf_scores = self.trans_fn(scores)
        return tf_scores, scores

def unittest():
    herceptin = FVTemplate(
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS',
        'DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK',
        'SRWGGDGFYAMDY',
        'EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYC',
        'WGQGTLVTVSS',
        'QDVNTA', 'QQHYTTPPT', 'SR', 'Y')

    aa_seqs = ['SRWGGDGFYAMDY', 'SRWGGDGFYADDY']

    MHC2affinity = MHCIIAffinity(template=herceptin)
    print(MHC2affinity(aa_seqs=aa_seqs))


if __name__ == '__main__':
    unittest()
    
