# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
import torch
from dataclasses import dataclass
from types import MethodType


logger = logging.getLogger(__name__)


LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    # was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    # was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # was this frame detected as repeating existing text content?
    repetition: bool
    # was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    # has inference reached the end of the text tokens? eg, this remains false if inference stops early
    complete: bool
    # approximate position in the text token sequence. Can be used for generating online timestamps.
    position: int


class AlignmentStreamAnalyzer:
    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        """
        Some transformer TTS models implicitly solve text-speech alignment in one or more of their self-attention
        activation maps. This module exploits this to perform online integrity checks which streaming.
        A hook is injected into the specified attention layer, and heuristics are used to determine alignment
        position, repetition, etc.

        NOTE: currently requires no queues.
        """
        # self.queue = queue
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, j-i)
        # self.alignment_bin = torch.zeros(0, j-i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None
        
        # Track generated tokens for repetition detection
        self.generated_tokens = []

        # Using `output_attentions=True` is incompatible with optimized attention kernels, so
        # using it for all layers slows things down too much. We can apply it to just one layer
        # by intercepting the kwargs and adding a forward hook (credit: jrm)
        self.last_aligned_attns = []
        for i, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self.last_aligned_attns += [None]
            self._add_attention_spy(tfmr, i, layer_idx, head_idx)

    def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
        """
        Adds a forward hook to a specific attention layer to collect outputs.
        """
        def attention_forward_hook(module, input, output):
            """
            See `LlamaAttention.forward`; the output is a 3-tuple: `attn_output, attn_weights, past_key_value`.
            NOTE:
            - When `output_attentions=True`, `LlamaSdpaAttention.forward` calls `LlamaAttention.forward`.
            - `attn_output` has shape [B, H, T0, T0] for the 0th entry, and [B, H, 1, T0+i] for the rest i-th.
            """
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                step_attention = output[1].cpu()  # (B, n_heads, T0, Ti)
                self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]  # (T0, Ti)

        target_layer = tfmr.layers[layer_idx].self_attn
        # Register hook and store the handle
        target_layer.register_forward_hook(attention_forward_hook)
        if hasattr(tfmr, 'config') and hasattr(tfmr.config, 'output_attentions'):
            self.original_output_attentions = tfmr.config.output_attentions
            tfmr.config.output_attentions = True

    def step(self, logits, next_token=None):
        """
        Emits an AlignmentAnalysisResult into the output queue, and potentially modifies the logits to force an EOS.
        """
        # extract approximate alignment matrix chunk (1 frame at a time after the first chunk)
        if not self.last_aligned_attns:
            return logits
            
        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0) # (N, N)
        i, j = self.text_tokens_slice
        
        if self.curr_frame_pos == 0:
            # first chunk has conditioning info, text tokens, and BOS token
            A_chunk = aligned_attn[j:, i:j].clone().cpu() # (T, S)
        else:
            # subsequent chunks have 1 frame due to KV-caching
            A_chunk = aligned_attn[:, i:j].clone().cpu() # (1, S)

        # TODO: monotonic masking; could have issue b/c spaces are often skipped.
        A_chunk[:, self.curr_frame_pos + 1:] = 0

        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        # update position
        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not(-4 < cur_text_posn - self.text_position < 7) 
        if not discontinuity:
            self.text_position = cur_text_posn

        # --- БЕЗПЕЧНІ ПЕРЕВІРКИ ГАЛЮЦИНАЦІЙ НА СТАРТІ ---
        false_start = False
        if not self.started:
            # Перевіряємо межі, щоб не було IndexError на дуже коротких послідовностях
            cond1 = A[-2:, -2:].max() > 0.1 if T >= 2 and S >= 2 else False
            cond2 = A[:, :4].max() < 0.5 if S >= 4 else True
            false_start = cond1 or cond2
            
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Is generation likely complete?
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # --- БЕЗПЕЧНИЙ РОЗРАХУНОК КЕРУВАННЯ EOS ---
        last_text_token_duration = 0
        if T > 15 and S >= 3:
            last_text_token_duration = A[15:, -3:].sum()

        long_tail = False
        if self.complete and self.completed_at is not None and S >= 3:
            # Перевіряємо, чи є дані після завершення
            if T > self.completed_at:
                long_tail = A[self.completed_at:, -3:].sum(dim=0).max() >= 5

        alignment_repetition = False
        if self.complete and self.completed_at is not None and S > 5:
            # ВИПРАВЛЕНО: додано перевірку на наявність рядків для аналізу
            sliced_A = A[self.completed_at:, :-5]
            if sliced_A.numel() > 0:
                alignment_repetition = sliced_A.max(dim=1).values.sum() > 5
        
        # Track generated tokens for repetition detection
        if next_token is not None:
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]
        
        # Check for excessive token repetition
        token_repetition = (
            len(self.generated_tokens) >= 3 and
            len(set(self.generated_tokens[-2:])) == 1
        )
        
        if token_repetition:
            repeated_token = self.generated_tokens[-1]
            logger.warning(f"🚨 Detected 2x repetition of token {repeated_token}")
            
        # Suppress EoS to prevent early termination
        if cur_text_posn < S - 3 and S > 5:
            logits[..., self.eos_idx] = -2**15

        # If a bad ending is detected, force emit EOS
        if long_tail or alignment_repetition or token_repetition:
            logger.warning(f"forcing EOS token, {long_tail=}, {alignment_repetition=}, {token_repetition=}")
            # Використовуємо велике значення для стабільності
            logits = torch.full_like(logits, -1e4)
            logits[..., self.eos_idx] = 1e4

        self.curr_frame_pos += 1
        return logits
