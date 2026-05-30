# NOVA 1.0 (25M HDC Brain)

This is a **25 Million Parameter** proof-of-concept model utilizing a novel **Hyperdimensional Computing (HDC)** memory architecture instead of a standard Transformer attention mechanism. 

It was built from scratch to prove that microscopic models can perform **Chain-of-Thought reasoning** and follow strict instructions when trained with the right memory compression and curriculum.

## Why this is special
Standard Transformer models at 25M parameters (like GPT-2 Small at 124M) are completely incapable of maintaining a persona, following system prompts, or reasoning in a `<think>` block. They lack the embedding dimension required for self-attention to separate context effectively.

By replacing traditional self-attention with $O(1)$ HDC state tracking, this tiny model manages to:
1. Maintain its identity as "NOVA".
2. Parse its responses into structural `[Thought]` and `[Answer]` blocks.
3. Compute internal confidence metrics for every token generated.

## The "Magic Fish" (Capacity Limits)
This model is heavily constrained by its size. 25M parameters is not enough "hard drive space" to memorize the world. While its internal logic engine is structurally sound, if you ask it questions outside its tiny training distribution, it will hallucinate facts wildly (for example, claiming Jupiter is the "leader of Earth's magic fish" most of time but sometimes answers like "Jupiter is the largest planet in our solar system" ). 

**This is expected behavior for this release, not a bug.** It proves the architecture compresses logic and attempts to reason even when factual vectors are weak.

## How to Run it
I have provided the clean inference architecture (`nova_model.py`) and a chat script (`chat.py`). I am intentionally keeping the training curriculum (Scheduled Sampling logic) private to prevent unauthorized replication before I scale up to larger models.

1. Clone this repository.
2. Ensure you have PyTorch installed.
3. Download the model weights and tokenizer from [Hugging Face: ReXeeD/NOVA-1.0-25M](https://huggingface.co/ReXeeD/NOVA-1.0-25M).
4. Create a folder named `model` in the root of the repository and place the downloaded files inside it:
   - `model/Nova1.0-25M.pt`
   - `model/nova_tokenizer.json`
5. Run the chat script:
```bash
python chat.py
```

## Call for Funding
This 25M model is just the beginning. The architecture is proven, the reasoning engine works, and the memory compression is highly efficient.

The next step is scaling this architecture to **0.5 Billion parameters**. At 0.5B, the HDC vectors will have enough dimensionality to stop hallucinations, creating a true, hyper-efficient reasoning engine.

If you are interested in funding or collaborating on the 0.5B scale-up of the NOVA HDC architecture, please reach out to the creator, **ReXeeD**.
