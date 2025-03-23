import torch
import torch.nn.functional as F

class TemperatureSampleDecoder:
    def __init__(self, tau=1.0):
        self.tau = tau
        
    def __call__(self, logits):
        if self.tau == 0:
            next_token_id = torch.argmax(logits, dim=-1)
        else:
            logits = logits/self.tau
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token_id.squeeze(-1)
        return next_token_id

logits = torch.randn(4, 50257)
decoder = TemperatureSampleDecoder()
next_token_id = decoder(logits)
print(next_token_id)
