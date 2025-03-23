import torch
import torch.nn.functional as F
# TODO 需要修改
class BeamSearchDecoder:
    def __init__(self, beam_size=5, max_length=20, eos_token_id=2):
        """
        初始化 Beam Search 解码器
        
        参数:
        beam_size (int): beam 的大小，即每一步保留的候选序列数量
        max_length (int): 生成序列的最大长度
        eos_token_id (int): 结束符的 token id
        """
        self.beam_size = beam_size
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        
    def decode(self, initial_logits, get_next_logits_fn):
        """
        执行 beam search 解码
        
        参数:
        initial_logits: 初始状态的 logits，形状为 [batch_size, vocab_size]
        get_next_logits_fn: 函数，接收当前序列，返回下一步的 logits
        
        返回:
        最终选择的序列和对应的分数
        """
        batch_size = initial_logits.shape[0]
        vocab_size = initial_logits.shape[-1]
        device = initial_logits.device
        
        # 初始化
        log_probs = F.log_softmax(initial_logits, dim=-1)  # [batch_size, vocab_size]
        
        # 为每个 batch 获取 beam_size 个最高概率的 token
        scores, indices = log_probs.topk(self.beam_size, dim=-1)  # [batch_size, beam_size]
        
        # 初始化 beam 状态
        beams = indices.view(batch_size, self.beam_size, 1)  # [batch_size, beam_size, 1]
        beam_scores = scores.view(batch_size, self.beam_size)  # [batch_size, beam_size]
        
        # 记录哪些序列已经完成
        done_beams = torch.zeros((batch_size, self.beam_size), dtype=torch.bool, device=device)
        
        # 开始 beam search
        for step in range(1, self.max_length):
            all_candidates = []
            all_scores = []
            all_predecessors = []
            
            # 对每个 batch 的每个 beam 进行扩展
            for batch_idx in range(batch_size):
                if done_beams[batch_idx].all():  # 如果该 batch 的所有 beam 都已完成
                    continue
                    
                # 获取当前 batch 中未完成的 beam
                active_beam_idx = (~done_beams[batch_idx]).nonzero(as_tuple=True)[0]
                active_beams = beams[batch_idx, active_beam_idx]
                
                # 获取每个活跃 beam 的下一步 logits
                next_logits = []
                for beam_idx in active_beam_idx:
                    # 调用提供的函数获取下一步 logits
                    beam_seq = beams[batch_idx, beam_idx]
                    next_logit = get_next_logits_fn(beam_seq)
                    next_logits.append(next_logit)
                
                next_logits = torch.stack(next_logits)  # [active_beams, vocab_size]
                next_log_probs = F.log_softmax(next_logits, dim=-1)  # [active_beams, vocab_size]
                
                # 计算新的候选序列的分数
                # [active_beams, vocab_size]
                vocab_scores = beam_scores[batch_idx, active_beam_idx].unsqueeze(-1) + next_log_probs
                
                # 重塑以便于排序
                vocab_scores = vocab_scores.view(-1)  # [active_beams * vocab_size]
                
                # 选择 beam_size 个最高分数
                topk_scores, topk_indices = vocab_scores.topk(min(self.beam_size, len(vocab_scores)))
                
                # 计算这些 token 来自哪个 beam 和对应的词汇表索引
                beam_indices = topk_indices // vocab_size
                token_indices = topk_indices % vocab_size
                
                # 保存候选信息
                all_candidates.append(token_indices)
                all_scores.append(topk_scores)
                all_predecessors.append(beam_indices)
            
            # 更新 beams 和分数
            for batch_idx in range(batch_size):
                if done_beams[batch_idx].all():
                    continue
                
                # 获取新的 token
                new_tokens = all_candidates[batch_idx].unsqueeze(-1)  # [beam_size, 1]
                predecessor_ids = all_predecessors[batch_idx]  # [beam_size]
                
                # 更新分数
                beam_scores[batch_idx] = all_scores[batch_idx]
                
                # 构建新的序列
                prev_beams = beams[batch_idx]  # [beam_size, step]
                new_beams = torch.cat([prev_beams[predecessor_ids], new_tokens], dim=-1)
                beams[batch_idx] = new_beams
                
                # 检查哪些序列已经结束
                done_beams[batch_idx] = new_beams[:, -1] == self.eos_token_id
            
            # 如果所有 batch 的所有 beam 都已完成，提前结束
            if done_beams.all():
                break
        
        # 选择每个 batch 中分数最高的序列
        best_indices = beam_scores.argmax(dim=-1)  # [batch_size]
        best_seqs = torch.stack([beams[i, idx] for i, idx in enumerate(best_indices)])
        best_scores = torch.stack([beam_scores[i, idx] for i, idx in enumerate(best_indices)])
        
        return best_seqs, best_scores

# 创建 Beam Search 解码器
beam_search = BeamSearchDecoder(beam_size=3, max_length=10, eos_token_id=2)

# 模拟词汇表大小和批次大小
vocab_size = 1000
batch_size = 2

# 生成随机的初始 logits
initial_logits = torch.randn(batch_size, vocab_size)

# 定义一个函数，根据当前序列生成下一步的 logits
# 在实际应用中，这个函数会使用模型来生成 logits
def get_next_logits_fn(sequence):
    """
    生成随机的下一步 logits，但保持一定的连贯性
    
    参数:
    sequence: 当前序列 [seq_len]
    
    返回:
    下一步的 logits [vocab_size]
    """
    # 使用最后一个 token 作为种子来生成一些偏好
    last_token = sequence[-1].item()
    
    # 创建随机 logits
    logits = torch.randn(vocab_size)
    
    # 为了模拟序列的连贯性，让某些 token 更可能跟随当前 token
    # 例如，如果最后一个 token 是 x，那么 x+1, x+2, x-1 可能更有可能出现
    preferred_tokens = [(last_token + i) % vocab_size for i in range(-2, 3)]
    for token in preferred_tokens:
        if 0 <= token < vocab_size:
            logits[token] += 2.0  # 增加偏好 token 的概率
    
    # 增加结束符的概率，随着序列变长
    if len(sequence) > 5:
        logits[2] += (len(sequence) - 5) * 0.5  # EOS token 的概率随序列长度增加
    
    return logits

# 执行 beam search
best_sequences, best_scores = beam_search.decode(initial_logits, get_next_logits_fn)

print("Beam Search 结果:")
for i in range(batch_size):
    print(f"批次 {i}:")
    print(f"  最佳序列: {best_sequences[i].tolist()}")
    print(f"  分数: {best_scores[i].item():.4f}")