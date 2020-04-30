import torch
import torch.nn as nn
START_TAG = '<START>'
STOP_TAG = '<STOP>'

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype= torch.long)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class CRF(nn.Module):
    def __init__(self, tag_to_idx):
        super(CRF, self).__init__()
        self.tag_to_idx = tag_to_idx
        self.tag_size = len(tag_to_idx)

        # transitions *to* i *from* j
        self.transitions = nn.Parameter(torch.randn(self.tag_size,self.tag_size))
    
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_idx[START_TAG]],dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i+1], tags[i]] + feats[tags[i + 1]]
        score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[-1]]
        return score
    def forward(self, feats, tags):
        # do the foward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tag_size), -10000.)
        # START_TAG has all of the score
        init_alphas[0][self.tag_to_idx[START_TAG]] = 0.

        # wrap in a variable so that we will get automatic backprop

        forward_var = init_alphas

        # iterate through the sentence
        for feat in feats:
            alphas_t = [] # the forward tensors at this timestep
            for next_tag in range(self.tag_size):
                
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_size)

                trans_score = self.transitions[next_tag].view(1, -1)

                next_tag_var = forward_var + trans_score + emit_score

                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1,-1)
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        forward_socre = log_sum_exp(terminal_var)
        gold_score = self._score_sentence(feats, tags)

        return forward_socre - gold_score

    def viterbi_decode(self, feats):
        pass