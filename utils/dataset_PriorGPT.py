import logging
import torch
from torch.utils.data import TensorDataset
import numpy as np

logger = logging.getLogger(__name__)

class AASeqDictionary_con(object):
    """
    A fixed dictionary for protein sequences.
    Enables sequence<->token conversion.
    With a space:0 for padding, B:1 as the start token and end_of_line \n:2 as the stop token.
    """
    PAD, BEGIN, END = ' ', 'B', '\n'

    def __init__(self):
        self.char_idx = {self.PAD: 0, self.BEGIN: 1, self.END: 2, 'A': 3, 'R': 4, 'N': 5, 'D': 6, 'C': 7, 'E': 8,
                         'Q': 9, 'G': 10, 'H': 11, 'I': 12, 'L': 13, 'K': 14, 'M': 15, 'F': 16, 'P': 17, 'S': 18,
                         'T': 19, 'W': 20, 'Y': 21, 'V': 22, 'X': 23,
                         'good_FvNetCharge':24, 'bad_FvNetCharge':25,  'good_FvCSP':26, 'bad_FvCSP':27,
                         'good_HISum':28, 'bad_HISum':29
                         }  # X for unknown AA
        self.idx_char = {v: k for k, v in self.char_idx.items()}

    def get_char_num(self) -> int:
        """
        Returns:
            number of characters in the alphabet
        """
        return len(self.idx_char)

    @property
    def begin_idx(self) -> int:
        return self.char_idx[self.BEGIN]

    @property
    def end_idx(self) -> int:
        return self.char_idx[self.END]

    @property
    def pad_idx(self) -> int:
        return self.char_idx[self.PAD]

    def matrix_to_seqs(self, array):
        """
        Converts a matrix of indices into their Sequence representations
        Args:
            array: torch tensor of indices, one sequence per row

        Returns: a list of Sequence, without the termination symbol
        """
        seqs_strings = []
        idxs=[]
        idx=0

        for row in array:
            predicted_chars = []
            flag=0
            for j in row:
                if j.item()==1:
                    continue
                elif j.item()==2:
                    break
                elif j.item()<2 or j.item()>23:
                    flag=1
                    break
                next_char = self.idx_char[j.item()]
                predicted_chars.append(next_char)
            if flag==0 and len(predicted_chars)>=12:
                seq = ''.join(predicted_chars)
                seqs_strings.append(seq)
                idxs.append(idx)
            idx+=1

        return seqs_strings,idxs
    
    def matrix_to_seqs_final(self, array,ori_idxs):
        """
        Converts a matrix of indices into their Sequence representations
        Args:
            array: torch tensor of indices, one sequence per row

        Returns: a list of Sequence, without the termination symbol
        """
        seqs_strings = []
        idxs=[]
        idx=0

        for row in array:
            predicted_chars = []
            flag=0
            for j in row:
                if j.item()==1:
                    continue
                elif j.item()==2:
                    break
                elif j.item()<2 or j.item()>23:
                    continue
                next_char = self.idx_char[j.item()]
                predicted_chars.append(next_char)
            if len(predicted_chars)==13:
                seq = ''.join(predicted_chars)
                seqs_strings.append(seq)
                idxs.append(ori_idxs[idx].item())
            idx+=1

        return seqs_strings,idxs

    def seqs_to_matrix(self, seqs, max_len=100):
        """
        Converts a list of seqs into a matrix of indices

        Args:
            seqs: a list of Sequence, without the termination symbol
            max_len: the maximum length of seqs to encode, default=100

        Returns: a torch tensor of indices for all the seqs
        """
        batch_size = len(seqs)
        # seqs = [self.BEGIN + seq + self.END for seq in seqs]
        idx_matrix = torch.zeros((batch_size, max_len))
        for i, seq in enumerate(seqs):
            enc_seq = seq
            for j in range(max_len):
                if j >= len(enc_seq):
                    break
                idx_matrix[i, j] = self.char_idx[enc_seq[j]]

        return idx_matrix.to(torch.int64)


class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores.
       Used to train agent.
       """
    def __init__(self, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.sd = AASeqDictionary_con()  # using to replace voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory)>self.max_size:
            # Remove duplicates
            idxs, seqs = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in seqs:
                    idxs.append(i)
                    seqs.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key = lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]
            logger.info("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores/np.sum(scores))  # random sampling
            sample = [self.memory[i] for i in sample]
            seqs = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        idx_matrix = self.sd.seqs_to_matrix(seqs,max_len=13)
        return idx_matrix, np.array(scores), np.array(prior_likelihood)

    def initiate_from_file(self, fname, scoring_function, Prior):
        """Adds experience from a file with Seqs
           Needs a scoring function and an RNN to score the sequences.
           Using this feature means that the learning can be very biased
           and is typically advised against."""
        with open(fname, 'r') as f:
            seqs = []
            for line in f:
                seq = line.split()[0]
                if Is_valid_seq(seq):
                    seqs.append(seq)
        scores = scoring_function(seqs)
        idx_matrix = self.sd.seqs_to_matrix(seqs)
        prior_likelihood, _ = Prior.likelihood(idx_matrix.long())  # Need to update
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(seqs, scores, prior_likelihood)
        self.add_experience(new_experience)

    def print_memory(self, path):
        """Prints the memory."""
        print("\n" + "*" * 80 + "\n")
        print("         Best recorded Seqs: \n")
        print("Score     Prior log P     Seqs\n")
        with open(path, 'w') as f:
            f.write("Seqs Score PriorLogP\n")
            for i, exp in enumerate(self.memory[:100]):
                if i < 50:
                    print("{:4.2f}   {:6.2f}        {}".format(exp[1], exp[2], exp[0]))
                    f.write("{} {:4.2f} {:6.2f}\n".format(*exp))
        print("\n" + "*" * 80 + "\n")

    def __len__(self):
        return len(self.memory)



def remove_duplicates(list_with_duplicates,valid_idx):
    """
    Removes the duplicates and keeps the ordering of the original list.
    For duplicates, the first occurrence is kept and the later occurrences are ignored.

    Args:
        list_with_duplicates: list that possibly contains duplicates

    Returns:
        A list with no duplicates.
    """

    unique_set = set()
    unique_list = []
    unique_idx = []
    i=0
    for element in list_with_duplicates:
        if element not in unique_set:
            unique_set.add(element)
            unique_list.append(element)
            unique_idx.append(valid_idx[i])
        i+=1

    return unique_list,unique_idx

def condition_convert(con_df):
    con_df = con_df.copy() 

    # convert to 0, 1
    con_df.loc[con_df['raw_FvNetCharge'] < 6.2 , 'raw_FvNetCharge'] = 1
    con_df.loc[con_df['raw_FvNetCharge'] >= 6.2 , 'raw_FvNetCharge'] = 0
    con_df.loc[con_df['raw_FvCSP'] <= 6.61, 'raw_FvCSP'] = 0
    con_df.loc[con_df['raw_FvCSP'] > 6.61, 'raw_FvCSP'] = 1
    con_df.loc[(con_df['raw_HISum'] >= 0) & (con_df['raw_HISum'] <= 4.0), 'raw_HISum'] = 1
    con_df.loc[(con_df['raw_HISum'] < 0) | (con_df['raw_HISum'] > 4.0), 'raw_HISum'] = 0

    # convert to token
    con_df.loc[con_df['raw_FvNetCharge'] == 1, 'raw_FvNetCharge'] = 'good_FvNetCharge'
    con_df.loc[con_df['raw_FvNetCharge'] == 0, 'raw_FvNetCharge'] = 'bad_FvNetCharge'
    con_df.loc[con_df['raw_FvCSP'] == 1, 'raw_FvCSP'] = 'good_FvCSP'
    con_df.loc[con_df['raw_FvCSP'] == 0, 'raw_FvCSP'] = 'bad_FvCSP'
    con_df.loc[con_df['raw_HISum'] == 1, 'raw_HISum'] = 'good_HISum'
    con_df.loc[con_df['raw_HISum'] == 0, 'raw_HISum'] = 'bad_HISum'
    return con_df


def load_seqs_from_list_con(df, column_cdrs,column_cons,rm_duplicates=True, max_len=100):
    """
    Given a list of Sequence strings, provides a zero padded NumPy array
    with their index representation. Sequences longer than `max_len` are
    discarded. The final array will have dimension (all_valid_seqs, max_len+2)
    as a beginning and end of sequence tokens are added to each string.

    Args:
        seqs_list: a list of Sequence strings
        rm_duplicates: bool if True return remove duplicates from final output. Note that if True the length of the
          output does not equal the size of the input  `seqs_list`. Default True
        max_len: dimension 1 of returned array, sequences will be padded

    Returns:
        sequences: list a numpy array of Sequence character indices
    """
    sd = AASeqDictionary_con()
    seqs_list = df[column_cdrs].values.tolist()
    cons_list = condition_convert(df[column_cons]).values.tolist()

    # filter valid seqs strings
    valid_seqs = []
    valid_mask = [False] * len(seqs_list)
    valid_idx=[]
    for i, s in enumerate(seqs_list):
        s = s.strip()
        if len(s) <= max_len and 'X' not in s:
            valid_seqs.append(s)
            valid_mask[i] = True
            valid_idx.append(i)

    if rm_duplicates: 
        unique_seqs,unique_idxs = remove_duplicates(valid_seqs,valid_idx)
    else:
        unique_seqs = valid_seqs
        unique_idxs = valid_idx
    # unique_seqs=seqs_list
    # unique_idxs=[i for i in range(len(unique_seqs))]

    # max len + two chars for start token 'Q' and stop token '\n'
    max_seq_len = max_len + 2 + 3
    num_seqs = len(unique_seqs)
    logger.info(f'Number of sequences: {num_seqs}, max length: {max_len}')

    # allocate the zero matrix to be filled
    sequences = np.zeros((num_seqs, max_seq_len), dtype=np.int32)
    for i, idx in enumerate(unique_idxs):
        seq=unique_seqs[i]
        enc_seq=[]
        enc_seq.append(sd.BEGIN)
        for x in cons_list[idx]:
            enc_seq.append(x)
        for x in seq:
            enc_seq.append(x)
        enc_seq.append(sd.END)
        for c in range(len(enc_seq)):
            try:
                sequences[i, c] = sd.char_idx[enc_seq[c]]
            except KeyError as e:
                logger.info(f'KeyError: {seq}, key: {i}, {enc_seq[c]}')
    return sequences

def get_tensor_dataset(numpy_array):
    """
    Gets a numpy array of indices, convert it into a Torch tensor,
    divided it into inputs and targets and wrap it into a TensorDataset

    Args:
        numpy_array: to be converted

    Returns: a TensorDataset
    """

    tensor = torch.from_numpy(numpy_array).long() 
    
    inp = tensor[:, :-1]
    target = tensor[:, 1:]

    return TensorDataset(inp, target)



def Is_valid_seq(seq):
    if 'X' in seq:  # Ignore seq with unknown AAs
        return False
    return True


def rnn_start_token_vector(batch_size, device='cpu'):
    """
    Returns a vector of start tokens. This vector can be used to start sampling a batch of Sequence strings.

    Args:
        batch_size: how many Sequence will be generated at the same time
        device: cpu | cuda

    Returns:
        a tensor (batch_size x 1) containing the start token
    """
    sd = AASeqDictionary_con()
    return torch.LongTensor(batch_size, 1).fill_(sd.begin_idx).to(device)
