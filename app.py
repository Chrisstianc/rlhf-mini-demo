import os, json, re, torch
import gradio as gr
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==== CONFIG ====
BASE_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_REPO = "christiancadena/qwen2.5-0.5b-dpo-lora"  # <-- your adapter repo id

# CPU-friendly settings
torch.set_num_threads(1)

# ==== Download adapter locally & sanitize its config (fixes weird keys) ====
adapter_path = snapshot_download(repo_id=ADAPTER_REPO)

cfg_path = os.path.join(adapter_path, "adapter_config.json")
if os.path.exists(cfg_path):
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        # Only keep keys PEFT's LoraConfig actually knows about
        allowed = {
            "peft_type",
            "auto_mapping",
            "base_model_name_or_path",
            "bias",
            "inference_mode",
            "modules_to_save",
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "task_type",
        }
        cleaned = {k: v for k, v in cfg.items() if k in allowed}
        if cleaned != cfg:
            with open(cfg_path, "w") as f:
                json.dump(cleaned, f)
    except Exception as e:
        # If anything goes wrong, continue; PEFT may still load if file is fine
        print("Warning: could not sanitize adapter_config.json:", e)

# ==== Load base model & tokenizer on CPU ====
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Keep everything on CPU to avoid accelerate offload issues
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,   # CPU-safe dtype
    low_cpu_mem_usage=True,
)
base.to("cpu")

# Attach LoRA
ft = PeftModel.from_pretrained(base, adapter_path)
ft.to("cpu")
ft.eval()

def generate_reply(prompt, temperature=0.7, max_new_tokens=200):
    if not prompt or not prompt.strip():
        return "Please enter a prompt."
    msgs = [
        {"role": "system", "content": "You are a helpful, clear assistant."},
        {"role": "user",   "content": prompt.strip()},
    ]
    inputs = tok.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    )
    inputs = inputs.to("cpu")
    attn = (inputs != tok.pad_token_id)

    with torch.no_grad():
        out = ft.generate(
            inputs,
            attention_mask=attn,
            do_sample=True,
            temperature=float(temperature),
            top_p=0.9,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    # Trim the prompt from the beginning if present
    if prompt in text:
        text = text.split(prompt, 1)[-1].strip()
    return text

# ==== Gradio UI ====
with gr.Blocks() as demo:
    gr.Markdown("# RLHF-mini: Qwen 0.5B + LoRA (DPO) on 41 pairs")
    gr.Markdown(
        "Tiny LoRA adapter (~18 MB) trained with DPO to improve tone/clarity/safety. "
        "Runs on CPU."
    )

    with gr.Row():
        inp = gr.Textbox(lines=4, label="Prompt", value="Rewrite this to be friendlier but still professional: 'Your payment is overdue.'")
    with gr.Row():
        temp = gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature")
        max_toks = gr.Slider(32, 512, value=200, step=8, label="Max new tokens")
    btn = gr.Button("Generate")
    out = gr.Textbox(lines=8, label="Output")

    btn.click(fn=generate_reply, inputs=[inp, temp, max_toks], outputs=out)

demo.launch()
