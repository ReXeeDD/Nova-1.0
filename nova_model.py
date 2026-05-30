"""
ΣNOVA v7: O(N) Recurrent Latent-Reasoning Architecture (~28M Params)

Key innovations over v6:
  - Multi-Timescale HDC Memory (Fast/Slow) with τ-gate routing
  - Sinusoidal positional encoding (0 learnable params)
  - Frozen orthogonal HDC projections (VSA trick, saves 8.3M params)
  - JEPA latent predictive coding with EMA target encoder
  - Per-layer learnable decay rates
  - RMSNorm throughout
  - Tied input/output embeddings

Loop structure: time-outer, layer-inner (causal synchronization).
Only layer 0's projections are vectorized across the sequence dimension.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# =============================================================================
# Utilities
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale


class NovaTokenizer:
    """
    Wrapper that handles both custom BPE tokenizer (from HuggingFace tokenizers lib)
    and the fallback tiktoken tokenizer, providing a unified encode/decode interface.
    """
    def __init__(self, tokenizer_path=None):
        self.tokenizer_path = tokenizer_path
        self.is_custom = False

        if tokenizer_path and os.path.exists(tokenizer_path):
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self.vocab_size = self._tokenizer.get_vocab_size()
            self.is_custom = True
            print(f"  Loaded custom tokenizer: {self.vocab_size:,} tokens from {tokenizer_path}")
        else:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            self.vocab_size = 100277
            if tokenizer_path:
                print(f"  Warning: Custom tokenizer not found at {tokenizer_path}, falling back to cl100k_base")
            else:
                print(f"  Using default tokenizer: cl100k_base ({self.vocab_size:,} tokens)")

    def encode(self, text: str) -> list:
        if self.is_custom:
            return self._tokenizer.encode(text).ids
        else:
            return self._tokenizer.encode(text)

    def decode(self, ids: list) -> str:
        if self.is_custom:
            return self._tokenizer.decode(ids)
        else:
            return self._tokenizer.decode(ids)


# =============================================================================
# ΣNOVA v7 Model
# =============================================================================

class NovaGPUv7(nn.Module):
    """
    SigmaNOVA v7: O(N) Recurrent Latent-Reasoning Architecture (~28M Params)

    Includes 0-Parameter Sinusoidal Positions and Frozen Orthogonal HDC Projections.
    Loop structure: time-outer, layer-inner.
    """

    EOS_TOKEN = "<|end|>"
    SEP_TOKEN = "<|sep|>"

    def __init__(self, vocab_size=8192, hidden_dim=512, hdc_dim=4096, num_layers=4,
                 lr=1e-3, tokenizer_path=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.hdc_dim = hdc_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer
        self.tokenizer = NovaTokenizer(tokenizer_path)
        self.tokenizer_path = tokenizer_path

        # Override vocab_size if tokenizer provides one
        if tokenizer_path and self.tokenizer.is_custom:
            self.vocab_size = self.tokenizer.vocab_size
            vocab_size = self.vocab_size

        # =====================================================================
        # 1. Hypersphere Embeddings (Tied with output head)
        # =====================================================================
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight)
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=1)

        # =====================================================================
        # 2. Sinusoidal Positional Encoding (0 learnable parameters)
        #    Stored as a buffer: saved with model but never updated by optimizer.
        # =====================================================================
        position_tensor = torch.zeros(65536, hidden_dim)
        pos_idx = torch.arange(0, 65536, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        position_tensor[:, 0::2] = torch.sin(pos_idx * div_term)
        position_tensor[:, 1::2] = torch.cos(pos_idx * div_term)
        self.register_buffer('position_encoder', position_tensor)

        # =====================================================================
        # 3. Frozen Orthogonal HDC Projections (VSA Trick)
        #    Initialized orthogonally and frozen. Perfectly preserves HDC
        #    mathematical orthogonality. Saves ~8.3M trainable parameters.
        # =====================================================================
        self.to_hdc_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hdc_dim, bias=False) for _ in range(num_layers)
        ])
        for layer in self.to_hdc_layers:
            nn.init.orthogonal_(layer.weight)
            layer.requires_grad_(False)

        # =====================================================================
        # 4. Per-Layer τ-Gate, Combine, and FFN Modules
        # =====================================================================
        self.tau_nets = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_layers)
        ])
        self.combine_layers = nn.ModuleList([
            nn.Linear(hdc_dim * 2, hidden_dim) for _ in range(num_layers)
        ])
        self.combine_norms = nn.ModuleList([
            RMSNorm(hidden_dim) for _ in range(num_layers)
        ])

        # FFN Modules (2x Expansion: 512 -> 1024 -> 512)
        self.ffn_expands = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 2) for _ in range(num_layers)
        ])
        self.ffn_compresses = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            RMSNorm(hidden_dim) for _ in range(num_layers)
        ])

        # =====================================================================
        # 5. Per-Layer Learnable Decay Parameters
        #    Initialized in logit-space so sigmoid gives ~0.95 and ~0.995
        # =====================================================================
        self.alpha_fast = nn.ParameterList([
            nn.Parameter(torch.tensor(2.944)) for _ in range(num_layers)  # sigmoid -> 0.95
        ])
        self.alpha_slow = nn.ParameterList([
            nn.Parameter(torch.tensor(5.293)) for _ in range(num_layers)  # sigmoid -> 0.995
        ])

        # =====================================================================
        # 6. JEPA Prediction Head
        # =====================================================================
        self.jepa_predictor = nn.Linear(hidden_dim, hidden_dim)

        # =====================================================================
        # 7. Learnable Logit Scale (CLIP-style)
        # =====================================================================
        self.logit_scale = nn.Parameter(torch.tensor(math.log(15.0)))

        # =====================================================================
        # Move to device, setup optimizer and AMP
        # =====================================================================
        self.to(self.device)

        # AdamW with recurrence-optimized hyperparameters
        # Only optimize parameters that require gradients (excludes frozen HDC projections)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95)
        )

        # Mixed Precision
        self.use_amp = (self.device.type == 'cuda')
        self.grad_scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.current_step = 0

        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters:    {frozen_params:,} (HDC projections)")

    # =========================================================================
    # Memory Management
    # =========================================================================

    def init_memory(self, batch_size, device=None):
        """Initialize per-layer Fast/Slow HDC memory states to zeros."""
        if device is None:
            device = self.device
        fast_hdc = [torch.zeros(batch_size, self.hdc_dim, device=device)
                    for _ in range(self.num_layers)]
        slow_hdc = [torch.zeros(batch_size, self.hdc_dim, device=device)
                    for _ in range(self.num_layers)]
        return fast_hdc, slow_hdc

    @staticmethod
    def detach_memory(fast_hdc_list, slow_hdc_list):
        """Detach memory states for TBPTT chunk boundaries."""
        return (
            [f.detach() for f in fast_hdc_list],
            [s.detach() for s in slow_hdc_list]
        )

    # =========================================================================
    # JEPA EMA Target Encoder
    # =========================================================================

    def create_ema_target(self):
        """Create a deep copy of this model as the EMA target encoder."""
        target = copy.deepcopy(self)
        # Freeze the target encoder — it's only updated via EMA
        for param in target.parameters():
            param.requires_grad_(False)
        target.eval()
        return target

    def update_ema_target(self, target_encoder, momentum=0.99):
        """
        Call AFTER optimizer.step(), strictly inside a no_grad() block.
        Updates the target encoder's parameters via exponential moving average.
        """
        with torch.no_grad():
            for param_q, param_k in zip(self.parameters(), target_encoder.parameters()):
                param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(self, x_tokens, position_ids, fast_hdc_list, slow_hdc_list, pad_token_id=0):
        """
        Args:
            x_tokens:       (Batch, Seq_Len) - token IDs
            position_ids:   (Batch, Seq_Len) - absolute position IDs (for TBPTT)
            fast_hdc_list:  list of num_layers tensors, each (Batch, HDC_Dim)
            slow_hdc_list:  list of num_layers tensors, each (Batch, HDC_Dim)
            pad_token_id:   ID of the padding token to ignore in memory updates

        Returns:
            logits:           (Batch, Seq_Len, Vocab_Size)
            jepa_predictions: (Batch, Seq_Len-1, Hidden_Dim)
            hidden_states:    (Batch, Seq_Len, Hidden_Dim)
            fast_hdc_list:    updated fast memory states
            slow_hdc_list:    updated slow memory states
        """
        batch_size, seq_len = x_tokens.size()

        # Identify padding tokens to prevent them from corrupting the EMA memory
        is_pad = (x_tokens == pad_token_id).to(torch.float32).unsqueeze(-1)  # (B, T, 1)

        # Pre-compute per-layer alpha values (scalar per layer)
        alpha_fast_vals = [torch.sigmoid(self.alpha_fast[i]) for i in range(self.num_layers)]
        alpha_slow_vals = [torch.sigmoid(self.alpha_slow[i]) for i in range(self.num_layers)]

        # Pre-compute token and position embeddings for full sequence
        # (Batch, Seq_Len, Hidden_Dim)
        seq_token_emb = F.normalize(self.embedding(x_tokens), p=2, dim=-1)
        # Sinusoidal buffer slice (no learnable params)
        seq_pos_emb = F.normalize(self.position_encoder[position_ids], p=2, dim=-1)

        # ---- Vectorize layer 0 projections outside the time loop ----
        # Layer 0 input is static (token embeddings), so we can batch the heavy
        # Linear(512 -> 4096) across the full sequence in one GPU call.
        bound_seq_0 = seq_token_emb * seq_pos_emb             # (B, T, H) - positional binding
        bound_hdc_0 = self.to_hdc_layers[0](bound_seq_0)      # (B, T, HDC) - one batched GPU call
        taus_0 = torch.sigmoid(self.tau_nets[0](seq_token_emb))  # (B, T, 1)

        hidden_states = []

        # ---- Time-outer loop: all layers see token t before moving to t+1 ----
        for t in range(seq_len):
            current_input = seq_token_emb[:, t, :]  # (B, H)

            for layer_idx in range(self.num_layers):
                fast_hdc = fast_hdc_list[layer_idx]
                slow_hdc = slow_hdc_list[layer_idx]

                if layer_idx == 0:
                    # Use pre-computed projections (vectorized outside time loop)
                    b_tok_hdc = bound_hdc_0[:, t, :]    # (B, HDC)
                    tau = taus_0[:, t, :]                # (B, 1)
                else:
                    # Layers 1-3: inputs depend on previous layer's live output
                    # at this timestep. Cannot be pre-computed.
                    # No positional binding for deeper layers (content-only).
                    b_tok_hdc = self.to_hdc_layers[layer_idx](current_input)     # (B, HDC)
                    tau = torch.sigmoid(self.tau_nets[layer_idx](current_input))  # (B, 1)

                tau_expanded = tau.expand_as(b_tok_hdc)  # (B, 1) -> (B, HDC)

                # Multi-Timescale HDC Memory Updates
                af = alpha_fast_vals[layer_idx]
                asl = alpha_slow_vals[layer_idx]

                new_fast_hdc = af * fast_hdc + (1 - af) * b_tok_hdc
                new_slow_hdc = slow_hdc + (tau_expanded * (b_tok_hdc - slow_hdc) * (1 - asl))

                # Mask out padding tokens: if is_pad == 1, keep old memory.
                pad_mask = is_pad[:, t, :]  # (B, 1)
                fast_hdc = pad_mask * fast_hdc + (1 - pad_mask) * new_fast_hdc
                slow_hdc = pad_mask * slow_hdc + (1 - pad_mask) * new_slow_hdc

                # Write updated states back
                fast_hdc_list[layer_idx] = fast_hdc
                slow_hdc_list[layer_idx] = slow_hdc

                # Combine fast + slow into hidden dim
                combined_hdc = torch.cat([fast_hdc, slow_hdc], dim=-1)  # (B, HDC*2)
                h = self.combine_layers[layer_idx](combined_hdc)        # (B, H)

                # Residual from this layer's input.
                # At t=0, fast_hdc and slow_hdc are zeros so h is a projection of zeros.
                # This residual allows the token signal to survive early timesteps.
                h = h + current_input
                h = self.combine_norms[layer_idx](h)

                # FFN
                ffn_out = F.gelu(self.ffn_expands[layer_idx](h))
                ffn_out = self.ffn_compresses[layer_idx](ffn_out)
                current_input = self.ffn_norms[layer_idx](h + ffn_out)  # becomes input to next layer

            hidden_states.append(current_input)

        hidden_states = torch.stack(hidden_states, dim=1)  # (B, T, H)

        # JEPA: predict hidden state at t+1 from hidden state at t
        jepa_predictions = self.jepa_predictor(hidden_states[:, :-1, :])  # (B, T-1, H)

        # Tied weights output projection with learnable logit scale
        logits = F.linear(
            F.normalize(hidden_states, p=2, dim=-1),
            F.normalize(self.embedding.weight, p=2, dim=-1)
        )
        logits = logits * self.logit_scale.exp().clamp(max=100)

        return logits, jepa_predictions, hidden_states, fast_hdc_list, slow_hdc_list

    # =========================================================================
    # Generation (Inference)
    # =========================================================================

    @torch.no_grad()
    def generate(self, prompt: str, n_tokens: int = 50, temperature: float = 0.8,
                 top_k: int = 50, repetition_penalty: float = 1.3, return_confidence: bool = False):
        """Autoregressively generate text from a prompt, optionally returning average confidence."""
        self.eval()

        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            if return_confidence:
                return "", 0.0
            return ""

        fast_hdc, slow_hdc = self.init_memory(1)

        # Feed prompt through model to build memory state (use same precision as training!)
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            for i, tid in enumerate(prompt_ids):
                x = torch.tensor([[tid]], device=self.device)
                pos = torch.tensor([[i]], device=self.device)
                logits, _, _, fast_hdc, slow_hdc = self.forward(x, pos, fast_hdc, slow_hdc)

        # Generate new tokens
        generated_ids = []
        confidence_scores = []
        next_pos = len(prompt_ids)

        for _ in range(n_tokens):
            # Record raw confidence before any temperature/penalty flattening
            raw_probs = torch.softmax(logits[0, -1, :], dim=-1)
            confidence_scores.append(torch.max(raw_probs).item())
            
            # Use last logit to sample next token
            next_logits = logits[0, -1, :].clone()

            # --- Repetition Penalty ---
            # Penalize tokens that have already been generated
            if generated_ids and repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty

            # --- Temperature ---
            next_logits = next_logits / temperature
            # Removed torch.clamp: PyTorch softmax is mathematically stable and subtracts max(logits).
            # Clamping was flattening all top logits to exactly 80 for low temperatures,
            # destroying the argmax and turning Greedy search into random noise!

            # --- Top-K Filtering ---
            if top_k > 0 and top_k < next_logits.size(-1):
                top_k_vals, _ = torch.topk(next_logits, top_k)
                threshold = top_k_vals[-1]
                next_logits[next_logits < threshold] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)

            if probs.sum() > 0:
                chosen_id = torch.multinomial(probs, 1).item()
            else:
                chosen_id = torch.argmax(probs).item()

            generated_ids.append(chosen_id)

            # Check for EOS
            decoded = self.tokenizer.decode(generated_ids)
            if self.EOS_TOKEN in decoded:
                final_text = decoded.replace(self.EOS_TOKEN, "").strip()
                if return_confidence:
                    # Sort scores to find the lowest confidence moments (the factual bottlenecks)
                    confidence_scores.sort()
                    bottom_k = max(1, len(confidence_scores) // 10)
                    factual_conf = sum(confidence_scores[:bottom_k]) / bottom_k
                    return final_text, factual_conf
                return final_text

            # Feed generated token back
            x = torch.tensor([[chosen_id]], device=self.device)
            pos = torch.tensor([[next_pos]], device=self.device)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits, _, _, fast_hdc, slow_hdc = self.forward(x, pos, fast_hdc, slow_hdc)
            next_pos += 1

        final_text = self.tokenizer.decode(generated_ids).replace(self.EOS_TOKEN, "").strip()
        if return_confidence:
            confidence_scores.sort()
            bottom_k = max(1, len(confidence_scores) // 10)
            factual_conf = sum(confidence_scores[:bottom_k]) / bottom_k
            return final_text, factual_conf
        return final_text

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, path: str):
        state = {
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "current_step": self.current_step,
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "hdc_dim": self.hdc_dim,
            "num_layers": self.num_layers,
            "tokenizer_path": self.tokenizer_path,
        }
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        # Use strict=False to handle architecture upgrades (e.g., alpha_fast → fast_gate_nets)
        missing, unexpected = self.load_state_dict(state["model_state"], strict=False)
        if missing:
            print(f"  Note: {len(missing)} new params initialized fresh (architecture upgrade)")
        if unexpected:
            print(f"  Note: {len(unexpected)} old params skipped (removed from architecture)")
        try:
            self.optimizer.load_state_dict(state["optimizer_state"])
        except (ValueError, KeyError):
            print("  Note: Optimizer state incompatible, using fresh optimizer")
        self.current_step = state.get("current_step", 0)
