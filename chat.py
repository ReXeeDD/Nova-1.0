import sys
import os
import torch
import re

# Import the local standalone architecture
from nova_model import NovaGPUv7

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    GRAY = '\033[90m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def format_output(response_text):
    text = response_text.strip()
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    
    if think_match:
        think_content = think_match.group(1).strip()
        after_think = text[think_match.end():].strip()
        if '<think>' in after_think:
            after_think = after_think[:after_think.index('<think>')].strip()
        answer_content = after_think
    else:
        think_positions = [m.start() for m in re.finditer(r'<think>', text)]
        
        if len(think_positions) >= 2:
            think_start = think_positions[0] + len('<think>')
            think_end = think_positions[1]
            think_content = text[think_start:think_end].strip()
            remaining = text[think_end + len('<think>'):].strip()
            if '<think>' in remaining:
                remaining = remaining[:remaining.index('<think>')].strip()
            answer_content = remaining
        elif len(think_positions) == 1:
            think_content = text[think_positions[0] + len('<think>'):].strip()
            answer_content = ""
        else:
            print(f"\n{Colors.GREEN}{text}{Colors.RESET}\n")
            return
    
    think_content = think_content.replace('</think>', '').replace('<think>', '').strip()
    answer_content = answer_content.replace('</think>', '').replace('<think>', '').strip()
    
    if think_content:
        print(f"\n{Colors.GRAY}{Colors.BOLD}[Thought]{Colors.RESET}")
        print(f"{Colors.GRAY}{think_content}{Colors.RESET}")
    
    if answer_content:
        print(f"\n{Colors.GREEN}{Colors.BOLD}[Answer]{Colors.RESET}")
        print(f"{Colors.GREEN}{answer_content}{Colors.RESET}\n")
    elif think_content:
        print(f"\n{Colors.YELLOW}(Answer was embedded in thought process above){Colors.RESET}\n")
    else:
        print(f"\n{Colors.YELLOW}(No coherent output){Colors.RESET}\n")


def main():
    print(f"\n{Colors.CYAN}{'=' * 70}")
    print("  NOVA 1.0: 25M HDC Brain")
    print(f"{'=' * 70}{Colors.RESET}")

    tokenizer_path = r"model\nova_tokenizer.json"
    weight_path = r"model\Nova1.0-25M.pt"

    print(f"Loading tokenizer...")
    if not os.path.exists(tokenizer_path):
        print(f"{Colors.YELLOW}Error: Tokenizer not found at {tokenizer_path}{Colors.RESET}")
        return

    # Initialize model (25M parameters)
    model = NovaGPUv7(
        vocab_size=8192,
        hidden_dim=512,
        hdc_dim=4096,
        num_layers=4,
        tokenizer_path=tokenizer_path
    )

    if os.path.exists(weight_path):
        print(f"Loading weights: {weight_path}...")
        model.load(weight_path)
    else:
        print(f"{Colors.YELLOW}Error: Could not find weights at {weight_path}{Colors.RESET}")
        return

    model.eval()
    
    temperature = 0.1
    n_tokens = 150
    rep_penalty = 1.15

    print(f"\n{Colors.CYAN}Brain online! Type 'quit' to exit.{Colors.RESET}")
    
    while True:
        try:
            prompt = input(f"{Colors.BOLD}You > {Colors.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt:
            continue
        if prompt.lower() in ["quit", "exit"]:
            break

        formatted_prompt = f"{prompt} {NovaGPUv7.SEP_TOKEN}"
        
        try:
            raw_response, confidence = model.generate(
                formatted_prompt, 
                n_tokens=n_tokens, 
                temperature=temperature,
                repetition_penalty=rep_penalty,
                return_confidence=True
            )
            
            # Confidence Interceptor
            if confidence < 0.1:
                print(f"\n{Colors.GRAY}[Internal Confidence {confidence*100:.1f}% too low. Intercepting hallucination.]{Colors.RESET}")
                raw_response = "<think>\nMy average prediction confidence fell below 10%. I should not guess.\n</think>\nI am sorry, but I have very limited knowledge in this field."
                
            format_output(raw_response)

        except Exception as e:
            print(f"{Colors.YELLOW}Generation error: {e}{Colors.RESET}")

if __name__ == "__main__":
    main()
