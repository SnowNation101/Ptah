# Ptah: Building the Future of Research with Vision and Language

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2605.29861)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/SnowNation101/Ptah?style=flat\&logo=github\&color=lightblue)](https://github.com/SnowNation101/Ptah)


This repository contains the official implementation of our paper, **"Towards Verifiable Multimodal Deep Research: A Multi-Agent Harness for Interleaved Report Generation."**

**Authors:** Chenghao Zhang, Guanting Dong, Yufan Liu, Tong Zhao, Xiaoxi Li, and Zhicheng Dou

## Preparation

```bash
conda create -n ptah python=3.11
conda activate ptah
pip install -r requirements.txt
```

Required environment variables are loaded from `.env` when present:

```bash
SERPER_API_KEY=...
JINA_API_KEY=...
OPENAI_API_KEY=...
```

Jina Reader is called directly from the server through `https://r.jina.ai`. No local SSH tunnel or local Jina proxy is required.

## Serve Local Models

Start Qwen3 on CUDA 0,1:

```bash
bash scripts/serve_llm.sh
```

Start Qwen3-VL on CUDA 2,3:

```bash
bash scripts/serve_vlm.sh
```

Defaults:

```text
LLM: models/Qwen3-32B at http://localhost:8000/v1/
VLM: models/Qwen3-VL-32B-Instruct at http://localhost:8001/v1/
```

## Run Reports

Run a custom report:

```bash
bash scripts/run_custom.sh
```

Run Deep Consult tasks:

```bash
bash scripts/run_dc.sh
```

Run DeepResearch Bench tasks:

```bash
bash scripts/run_drb.sh
```

## Outputs

DC outputs:

```text
outputs/dc/report_<id>.json
outputs/dc/report_<id>.html
.cache/dc/question_<id>/
```

DRB outputs:

```text
outputs/drb/report_<id>.json
outputs/drb/report_<id>.html
.cache/drb/question_<id>/
```

Custom outputs:

```text
outputs/custom/report.json
outputs/custom/report.html
.cache/custom/
```

## Evaluation

### DeepConsult

Run DC evaluation:

```bash
bash scripts/eval_dc.sh
```

Run DC PtahEval:

```bash
bash scripts/eval_dc_ptaheval.sh
```

### DeepResearch Bench

Run DRB RACE:

```bash
bash scripts/eval_drb_race.sh
```

Run DRB FACT:

```bash
bash scripts/eval_drb_fact.sh
```


Run DRB PtahEval:

```bash
bash scripts/eval_drb_ptaheval.sh
```

## Citation

```bibtex
@article{zhang2026ptah,
  author       = {Chenghao Zhang and
                  Guanting Dong and
                  Yufan Liu and
                  Tong Zhao and
                  Xiaoxi Li and
                  Zhicheng Dou},
  title        = {Towards Verifiable Multimodal Deep Research: A Multi-Agent Harness
                  for Interleaved Report Generation},
  journal      = {CoRR},
  volume       = {abs/2605.29861},
  year         = {2026},
  url          = {https://doi.org/10.48550/arXiv.2605.29861},
  doi          = {10.48550/ARXIV.2605.29861},
  eprinttype   = {arXiv},
  eprint       = {2605.29861},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2605-29861.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
