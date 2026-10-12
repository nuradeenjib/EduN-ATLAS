EduN-ATLAS

A QLoRA adapter on [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) for Nigerian university Computer Science
multiple-choice questions aligned to the NUC CCMAS 2023 Computing curriculum (17 courses, 114 learning outcomes).
Entry to the National AI Innovation Challenge 2026, Academia and Research Track, Problem Statement 3.

- Adapter: https://huggingface.co/El-Deen/EduN-ATLAS-lora-v1 (access to the gated base model is required)
- Technical documentation: see `docs/`
- Method: LoRA rank 8, alpha 16, 20,971,520 trainable parameters (0.26%), 4-bit NF4 base, 1 epoch on 600 items, seed 42

Results (540-item test set, same prompt and scoring for both models)
| Slice | Items | Base | Adapted | Difference (95% CI) |
|---|---|---|---|---|
| All | 540 | 90.2% | 97.0% | +6.9 points (3.8 to 10.1) |
| In-scope LOs | 376 | 90.4% | 96.3% | +5.9 points (2.3 to 10.1) |
| Holdout LOs | 164 | 89.6% | 98.8% | +9.2 points (4.4 to 14.7) |

Paired test: 7 items only the base answered correctly, 44 only the adapter (exact McNemar p < 0.0001).
Confidence intervals come from a cluster bootstrap over learning outcomes.

Limitations
Items were drafted with LLM assistance and reviewed by a
lecturer. English only. Study and practice aid only for now; not for high-stakes assessment decisions.

Repository layout
- `app/app.py`: FastAPI service (`/health`, `/v1/answer`, `/docs`)
- `data/`: train, development and test CSVs
- `results/`: item-level predictions, summaries and the evaluation protocol
- `configs/`: training record, loss log, adapter configuration, environment and package versions
- `docs/`: technical documentation and validation report (PDF)
- `EduN_ATLAS.ipynb`: the Colab notebook used for the experiments

Quick start
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained("NCAIR1/N-ATLaS", quantization_config=bnb, device_map="auto")
tok = AutoTokenizer.from_pretrained("NCAIR1/N-ATLaS")
model = PeftModel.from_pretrained(base, "El-Deen/EduN-ATLAS-lora-v1")
```
Run the API: `python app/app.py` (needs a GPU and Hugging Face access to the base model).
Example request: `POST /v1/answer` with `{"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}}`.

Licences and attribution
Base model: follow the N-ATLaS licence. Curriculum wording: NUC CCMAS 2023 Computing, CC BY-SA 3.0 IGO.
Questions: No access tokens are stored in this repository.

Team
Nurudeen Jibrin
Team Lead 

Kaigama Kashim Kaigama
Postgraduate Student