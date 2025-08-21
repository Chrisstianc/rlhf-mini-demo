# RLHF-mini (Qwen 0.5B + LoRA DPO)

**What:** Fine-tuned **Qwen/Qwen2.5-0.5B-Instruct** with **LoRA + DPO** on **41 human preference pairs**.  
**Why:** Show a tiny, end-to-end RLHF workflow that improves instruction-following and safe refusals.  
**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/christiancadena/rlhf-mini-demo)  
**Adapter:** [HF model repo](https://huggingface.co/christiancadena/qwen2.5-0.5b-dpo-lora)

---

## TL;DR results
**Before → After** (short examples)

- **Email tone rewrite**  
  *Base:* “Your bill is past due.”  
  *Tuned:* “Could you please take care of the payment when you can?”

- **Unsafe request refusal**  
  *Base:* suggested shady steps.  
  *Tuned:* clear refusal + safe alternatives.

---

## How I built it
1. Wrote ~30–50 prompts (instruction, safety, tone).
2. Auto-generated A/B candidates, then **manually labeled** winners with a simple rubric.
3. Trained **LoRA** with **DPO** (tiny Colab run).
4. Deployed **Gradio** app to a Hugging Face Space.

---

## Run locally (demo)
```bash
pip install -r requirements.txt
python app.py


