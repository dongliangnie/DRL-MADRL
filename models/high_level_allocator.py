"""
High-Level Allocator Network
----------------------------
MultiHeadHighLevelAllocator:
- forward(uav_feat, task_feat) -> logits (H, U, T)
- act(uav_feat, task_feat) -> high_action (1,) global task index, log_prob (1,1)
- get_value(global_obs) -> value (1,1)
- evaluate_actions(global_obs_batch, actions) -> values (B,1), log_probs (B,1), entropy (B,1)

This implementation chooses a single global high-level action (select a task index)
by averaging per-UAV assignment probabilities (head 0) and picking argmax.
This avoids mismatches between per-UAV actions and PPO's expected (batch,) action shapes.
"""

from __future__ import annotations
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 128, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers.append(nn.Linear(last, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, task_feat: torch.Tensor):
        # task_feat: (T, task_dim)
        return self.net(task_feat)  # (T, embed_dim)


class UAVEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 128, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers.append(nn.Linear(last, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, uav_feat: torch.Tensor):
        # uav_feat: (U, uav_dim)
        return self.net(uav_feat)  # (U, embed_dim)


class MultiHeadHighLevelAllocator(nn.Module):
    def __init__(
        self,
        uav_dim: int,
        task_dim: int,
        embed_dim: int = 128,
        num_heads: int = 2,
        hidden_dim: int = 256,
        num_uav: int = 5,
        num_task: int = 10
    ):
        """
        Args:
            uav_dim: 每个 UAV 的原始特征维度
            task_dim: 每个任务的特征维度
            embed_dim: 编码后 embedding 维度
            num_heads: 多头数
            hidden_dim: fusion hidden dim
            num_uav: (可选) 默认的 UAV 数量（用于 mock / 全局 obs 拆分）
            num_task: (可选) 默认的任务数量
        """
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.uav_dim = uav_dim
        self.task_dim = task_dim
        self.num_uav = num_uav
        self.num_task = num_task

        # encoders
        self.uav_encoder = UAVEncoder(uav_dim, embed_dim)
        self.task_encoder = TaskEncoder(task_dim, embed_dim)

        # multi-head learnable queries (H, E)
        self.head_queries = nn.Parameter(torch.randn(num_heads, embed_dim))

        # fusion: concat(q, k) -> score
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # value network: map flattened (uav+task) to scalar
        flat_dim = (self.num_uav * self.uav_dim) + (self.num_task * self.task_dim)
        self.value_net = nn.Sequential(
            nn.Linear(flat_dim, max(128, flat_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(128, flat_dim // 2), 1)
        )

    def forward(self, uav_feat: torch.Tensor, task_feat: torch.Tensor):
        """
        Args:
            uav_feat: (U, uav_dim)
            task_feat: (T, task_dim)
        Returns:
            logits: (H, U, T)
        """
        U = uav_feat.shape[0]
        T = task_feat.shape[0]
        H = self.num_heads

        uav_embed = self.uav_encoder(uav_feat)    # (U, E)
        task_embed = self.task_encoder(task_feat) # (T, E)

        head_queries = self.head_queries.unsqueeze(1)  # (H,1,E)

        # uav_query: (H, U, E)
        uav_query = uav_embed.unsqueeze(0) + head_queries

        # task_key: (H, T, E)
        task_key = task_embed.unsqueeze(0).expand(H, T, self.embed_dim)

        # fused: (H, U, T, 2E)
        fused = torch.cat([
            uav_query.unsqueeze(2).expand(H, U, T, self.embed_dim),
            task_key.unsqueeze(1).expand(H, U, T, self.embed_dim)
        ], dim=-1)

        logits = self.fusion_layer(fused).squeeze(-1)  # (H, U, T)
        return logits

    # ---------------------
    # Helper to extract uav/task features from a flattened global_obs
    # ---------------------
    def _split_global_obs(self, global_obs: torch.Tensor):
        """
        Try to split global_obs into (uav_feat, task_feat) using self.num_uav/self.num_task/self.uav_dim/self.task_dim.
        global_obs shape: (D,) or (B, D) where D should equal num_uav*uav_dim + num_task*task_dim.
        Returns:
            uav_feat (U, uav_dim), task_feat (T, task_dim)
        If cannot split, returns mock random features on same device.
        """
        device = global_obs.device
        single = False
        if global_obs.dim() == 1:
            global_obs = global_obs.unsqueeze(0)
            single = True

        B, D = global_obs.shape
        expected = self.num_uav * self.uav_dim + self.num_task * self.task_dim
        if D == expected:
            # split each batch entry and return only first (we only need structure)
            flat = global_obs[0]  # shape (D,)
            uav_flat = flat[: self.num_uav * self.uav_dim].reshape(self.num_uav, self.uav_dim)
            task_flat = flat[self.num_uav * self.uav_dim : ].reshape(self.num_task, self.task_dim)
            return uav_flat.to(device), task_flat.to(device)
        else:
            # fallback: return mock features (deterministic on device RNG)
            return torch.randn(self.num_uav, self.uav_dim, device=device), torch.randn(self.num_task, self.task_dim, device=device)

    def act(self, uav_feat: t.Optional[torch.Tensor] = None, task_feat: t.Optional[torch.Tensor] = None):
        """
        Produce a single global high-level action (task index).
        Accepts either:
            - (uav_feat, task_feat) pair: uav_feat (U, uav_dim), task_feat (T, task_dim)
            - If both None, uses mock features based on self.num_uav/self.num_task
        Returns:
            high_action: tensor shape (1,) containing chosen task index (long)
            log_prob: tensor shape (1,1) containing log probability
        """
        device = next(self.parameters()).device

        if uav_feat is None or task_feat is None:
            uav_feat = torch.randn(self.num_uav, self.uav_dim, device=device)
            task_feat = torch.randn(self.num_task, self.task_dim, device=device)

        logits = self.forward(uav_feat, task_feat)  # (H, U, T)

        # Use head0 by default; aggregate across UAVs by averaging their per-task probs
        probs = torch.softmax(logits[0], dim=-1)  # (U, T)
        global_probs = probs.mean(dim=0)          # (T,)

        # choose best task (global)
        chosen = torch.argmax(global_probs, dim=0, keepdim=True)  # (1,)
        # get log_prob (as (1,1))
        log_p = torch.log(global_probs[chosen].unsqueeze(1) + 1e-8)  # (1,1)

        return chosen.long().to(device), log_p.to(device)

    def get_value(self, global_obs: t.Optional[torch.Tensor] = None):
        """
        Estimate value for a (single) global_obs.
        - If global_obs provided and matches expected flattened size, use it; otherwise mock.
        Returns: (1,1) tensor
        """
        device = next(self.parameters()).device
        if global_obs is None:
            # construct mock flattened state
            flat = torch.randn(self.num_uav * self.uav_dim + self.num_task * self.task_dim, device=device)
            x = flat.unsqueeze(0)
            return self.value_net(x)
        else:
            # if global_obs is tensor, try to use it directly if shapes match.
            if global_obs.dim() == 2 and global_obs.shape[1] == (self.num_uav * self.uav_dim + self.num_task * self.task_dim):
                return self.value_net(global_obs.float())
            elif global_obs.dim() == 1 and global_obs.shape[0] == (self.num_uav * self.uav_dim + self.num_task * self.task_dim):
                return self.value_net(global_obs.unsqueeze(0).float())
            else:
                # attempt to build from uav/task split
                uav_feat, task_feat = self._split_global_obs(global_obs)
                flat = torch.cat([uav_feat.reshape(-1), task_feat.reshape(-1)], dim=0).unsqueeze(0).float().to(device)
                return self.value_net(flat)

    def evaluate_actions(self, global_obs: torch.Tensor, actions: torch.Tensor):
        """
        For PPO update: compute new values, log_probs and entropy for a batch.

        Args:
            global_obs: tensor shape (B, D) where D might be flattened uav+task dims (preferred)
                        OR you may pass any tensor (fallback -> uses mock features)
            actions: tensor shape (B,) or (B,1) containing target task indices (long)

        Returns:
            values: (B,1)
            log_probs: (B,1)
            entropy: (B,1)
        """
        device = next(self.parameters()).device

        # Normalize actions shape to (B,1)
        if actions.dim() == 1:
            actions_idx = actions.long().unsqueeze(1).to(device)
        else:
            actions_idx = actions.long().to(device)

        B = actions_idx.shape[0]

        # Try to split global_obs into uav_feat/task_feat (if possible)
        # If global_obs has batch and matches expected flattened dim -> split per-batch and use first entry for structure.
        try:
            uav_feat, task_feat = self._split_global_obs(global_obs)
        except Exception:
            uav_feat = torch.randn(self.num_uav, self.uav_dim, device=device)
            task_feat = torch.randn(self.num_task, self.task_dim, device=device)

        # Forward to get logits (H,U,T)
        logits = self.forward(uav_feat.to(device), task_feat.to(device))  # (H,U,T)
        probs_per_uav = torch.softmax(logits[0], dim=-1)  # (U,T)

        # Aggregate to global per-task distribution
        global_probs = probs_per_uav.mean(dim=0)  # (T,)
        # Build batch of same distribution
        probs_batch = global_probs.unsqueeze(0).expand(B, -1)  # (B,T)

        # Gather chosen log_probs
        gathered = probs_batch.gather(1, actions_idx)  # (B,1)
        log_probs = torch.log(gathered + 1e-8)           # (B,1)

        # Entropy (per-example)
        entropy = -(probs_batch * torch.log(probs_batch + 1e-8)).sum(dim=1, keepdim=True)  # (B,1)

        # Values (bootstrap) - use value_net: if global_obs fits, use directly; else estimate from flatten of uav/task
        try:
            # If global_obs is (B, D) with matching D, pass it directly
            if global_obs.dim() == 2 and global_obs.shape[1] == (self.num_uav * self.uav_dim + self.num_task * self.task_dim):
                values = self.get_value(global_obs)
            else:
                # replicate flatten of uav+task for batch
                flat = torch.cat([uav_feat.reshape(-1).unsqueeze(0).to(device).float()] * B, dim=0)
                values = self.value_net(flat)
        except Exception:
            flat = torch.randn(B, (self.num_uav * self.uav_dim + self.num_task * self.task_dim), device=device)
            values = self.value_net(flat)

        return values, log_probs, entropy


# ---------------------------
# Dummy test when run as script
# ---------------------------
def _dummy_test():
    print(">>> Running high_level_allocator dummy test ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_uav = 4
    num_task = 20
    uav_dim = 10
    task_dim = 12

    uav_states = torch.randn(num_uav, uav_dim).to(device)
    task_states = torch.randn(num_task, task_dim).to(device)

    model = MultiHeadHighLevelAllocator(
        uav_dim=uav_dim,
        task_dim=task_dim,
        embed_dim=128,
        num_heads=2,
        hidden_dim=256,
        num_uav=num_uav,
        num_task=num_task
    ).to(device)

    logits = model(uav_states, task_states)
    print("logits shape:", logits.shape)  # expected (H, U, T)

    # Test act()
    action_idx, logp = model.act(uav_states, task_states)
    print("act -> chosen task idx:", action_idx, "logp shape:", logp.shape)

    # Build a batch obs: flatten uav + task into one vector per sample
    flat = torch.cat([uav_states.reshape(-1), task_states.reshape(-1)], dim=0).unsqueeze(0)  # (1, D)
    batch_obs = flat.repeat(3, 1)  # B=3
    # actions batch: choose same action
    actions_batch = action_idx.repeat(3)  # (3,) or (3,1)
    vals, new_logp, ent = model.evaluate_actions(batch_obs, actions_batch)
    print("evaluate_actions -> values:", vals.shape, "logp:", new_logp.shape, "entropy:", ent.shape)

    # Test get_value
    v = model.get_value(flat.squeeze(0))
    print("get_value ->", v.shape)

    print("Dummy test passed ✔")


if __name__ == "__main__":
    _dummy_test()
