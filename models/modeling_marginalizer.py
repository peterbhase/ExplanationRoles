import torch
from torch import nn


class Marginalizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.NLL = torch.nn.NLLLoss()

    def forward(self, logits, labels, retrieval_scores):
        '''
        logits : batch_size x k x num_labels representing p(y|z,x)
        retrieval_scores : batch_size x k scores representing p(z|x)
        - this computes the model p(y|x) = 1/k sum_z p(y|z,x)p(z|x)
        - k is k samples of the latent variable
        - returns NLL
        '''
        latent_logprobs = torch.nn.functional.log_softmax(retrieval_scores, dim=-1)
        label_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        # combine label and logprobs. use unsqueeze(-1) bc want latent_logprobs to extend ALONG num_classes dimension
        joint_logprobs = label_logprobs + latent_logprobs.unsqueeze(-1) # yields shape batchsize x k x num_classes.
        marginal_logprobs = torch.logsumexp(joint_logprobs, dim=1) # this is the actual marginalization over k samples. returns ln p(y|x)
        loss = self.NLL(marginal_logprobs, labels)
        return loss, marginal_logprobs, label_logprobs, latent_logprobs

