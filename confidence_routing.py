"""Training-only confidence-aware, cluster-local, collision-free top-k refinement.
Original manuscript eqs 14--16; explicit collision handling and train/eval policy.
"""
import torch
from torch import nn
import torch.nn.functional as F


@torch.no_grad()
def refine_selection(probs, original, clusters, tau, exploration):
    """Refine low-margin original slots; reserve all other current slots.
    No duplicate experts per token; slot cluster membership is invariant.
    Discrete sampling has no score-function gradient, as in hard top-k selection.
    """
    selected = original.clone()
    if exploration <= 0:
        return selected
    tokens, n = probs.shape
    rows = torch.arange(tokens,device=probs.device)
    for slot in range(original.shape[1]):
        expert = original[:,slot]
        same_cluster = clusters[None,:].eq(clusters[expert,None])
        local = probs * same_cluster
        local = local/local.sum(-1,keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)
        other = local.clone()
        other[rows,expert] = -1.
        margin = local[rows,expert] - other.max(-1).values
        low = margin < tau
        occupied = torch.zeros_like(same_cluster)
        occupied.scatter_(1,selected,True)
        occupied[rows,selected[:,slot]] = False
        candidates = local.masked_fill(occupied,0.)
        empty = candidates.sum(-1)<=0
        candidates[rows,expert] += empty.to(candidates.dtype)
        # The original expert at this slot is always eligible, so mass is positive.
        sampled = torch.multinomial(candidates,1).squeeze(1)
        explore = low & (torch.rand(tokens,device=probs.device) < exploration)
        selected[:,slot] = torch.where(explore,sampled,expert)
    return selected


class ConfidenceMoE(nn.Module):
    def __init__(self, original, clusters, tau=.2):
        super().__init__()
        self.gate = original.gate
        self.experts = original.experts
        self.num_experts = original.num_experts
        self.top_k = original.top_k
        self.norm_topk_prob = original.norm_topk_prob
        if hasattr(original,'shared_expert'):
            self.shared_expert = original.shared_expert
            self.shared_expert_gate = original.shared_expert_gate
        self.register_buffer('cluster_ids',torch.tensor(clusters,dtype=torch.long,
                             device=self.gate.weight.device),persistent=False)
        self.native_forward = type(original).forward
        self.tau = float(tau)
        self.exploration = 0.
        self.last_dispatch = None

    def forward(self, hidden_states):
        if not self.training or self.exploration <= 0:
            self.last_dispatch = None
            return self.native_forward(self,hidden_states)
        shape = hidden_states.shape
        h = hidden_states.reshape(-1,shape[-1])
        logits = self.gate(h)
        probs = F.softmax(logits,dim=-1,dtype=torch.float32)
        original = probs.topk(self.top_k,dim=-1).indices
        selected = refine_selection(probs.detach(),original,self.cluster_ids,self.tau,self.exploration)
        self.last_dispatch = selected.detach()
        weights = probs.gather(1,selected)
        if self.norm_topk_prob:
            weights = weights/weights.sum(-1,keepdim=True)
        weights = weights.to(h.dtype)
        result = torch.zeros_like(h)
        # Same sparse dispatch and original mixture-weight convention after refinement.
        for expert_id,expert in enumerate(self.experts):
            token,slot = torch.where(selected==expert_id)
            values = expert(h.index_select(0,token))*weights[token,slot,None]
            result.index_add_(0,token,values.to(result.dtype))
        if hasattr(self,'shared_expert'):
            result = result + torch.sigmoid(self.shared_expert_gate(h))*self.shared_expert(h)
        return result.reshape(shape),logits


def install(model,plan,tau):
    for row in plan['layers']:
        old_labels = row['labels']
        if old_labels is None or len(old_labels)!=row['num_old_experts']:
            raise ValueError('Confidence routing requires an explicit expert grouping.')
        labels = list(old_labels)+[old_labels[parent] for parent in row['parents']]
        layer = model.model.layers[row['layer']]
        if len(labels)!=len(layer.mlp.experts):
            raise ValueError('Expanded expert/group count mismatch.')
        layer.mlp = ConfidenceMoE(layer.mlp,labels,tau)


def set_progress(model,step,total):
    # Parameter-free linear decay apart from the original confidence threshold tau.
    rho = max(0.,1.-step/max(1,total))
    for layer in model.model.layers:
        if isinstance(layer.mlp,ConfidenceMoE):
            layer.mlp.exploration = rho
    return rho
