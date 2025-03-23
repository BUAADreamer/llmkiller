import torch
import torch.nn.functional as F

class TopPSampleDecoder:
    def __init__(self, top_p=0.95):
        self.top_p = top_p
        
    def __call__(self, logits):
        if self.top_p>=1:
            next_token_id = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idxs = torch.sort(probs, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            nucleus_mask = cumsum_probs<=self.top_p
            nucleus_mask[:, 0]=True
            filtered_probs = sorted_probs * nucleus_mask
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            sample_idxs = torch.multinomial(filtered_probs, num_samples=1)
            next_token_id = torch.gather(sorted_idxs, dim=-1, index=sample_idxs).squeeze(-1)
            
        return next_token_id

logits = torch.randn(4, 50257)
decoder = TopPSampleDecoder()
next_token_id = decoder(logits)
print(next_token_id)           
        