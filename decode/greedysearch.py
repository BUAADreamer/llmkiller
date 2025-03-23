import torch

class GreedySearchDecoder:
    def __init__(self):
        pass
    
    def __call__(self, logits):
        next_token_id = torch.argmax(logits, dim=-1)
        return next_token_id

logits = torch.randn(4, 50257)
decoder = GreedySearchDecoder()
next_token_id = decoder(logits)
print(next_token_id)
