# RLHF-mini: Qwen 0.5B + LoRA (DPO) on 41 pairs

**Live demo:** https://huggingface.co/spaces/christiancadena/rlhf-mini-demo  
**Adapter weights:** https://huggingface.co/christiancadena/qwen2.5-0.5b-dpo-lora

Tiny end-to-end RLHF-style project:
- I generated A/B answers for ~41 prompts, labeled winners with a simple rubric, and trained a **LoRA** adapter with **DPO** on **Qwen/Qwen2.5-0.5B-Instruct**.
- The Space loads the base model + my LoRA and shows “before vs after” on real prompts.

## Quickstart (local)
```bash
pip install -r requirements.txt
python app.py

