import torch
import torch.nn.functional as F

class TopKSampleDecoder:
    def __init__(self, top_k=10):
        self.top_k = top_k
        
    def __call__(self, logits):
        vocab_size = logits.shape[-1]
        if self.top_k>=vocab_size:
            next_token_id = torch.argmax(logits, dim=-1)
        else:
            top_k_logits, top_k_idxs = torch.topk(logits, k=self.top_k, dim=-1)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            sampled_idxs = torch.multinomial(top_k_probs, num_samples=1) # [bs,1]
            next_token_id = torch.gather(top_k_idxs, dim=-1, index=sampled_idxs).squeeze(-1)
        return next_token_id

logits = torch.randn(4, 50257)
decoder = TopKSampleDecoder()
next_token_id = decoder(logits)
print(next_token_id)           
        