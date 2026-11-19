# 🎓 ssu.vshtani26 — Coursework & Internship

[![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-NCF-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![VK-LSVD](https://img.shields.io/badge/Dataset-VK--LSVD-0077FF?logo=vk&logoColor=white)](https://huggingface.co/datasets/deepvk/VK-LSVD)
[![Bruno](https://img.shields.io/badge/Bruno-API%20Tests-orange)](https://www.usebruno.com/)

[🇷🇺 Русский](README.md) · 🇬🇧 English

University projects by **Danila Fedotov**, a Computer Science & Engineering student at **SSU**.

| Folder | Description | Stack |
|--------|-------------|-------|
| [`cursovaya26/`](cursovaya26/) | 📊 Coursework: comparing recommendation methods on a VKontakte dataset | Python · scikit-surprise · PyTorch |
| [`prakt26/`](prakt26/) | 🧪 Reksoft internship: databases & API testing with Bruno | Bruno · OpenAPI · JavaScript |

> 🔗 The test collection also lives in a standalone repository:
> **[bruno-test-praktika](https://github.com/TWOINTHEPINKY/bruno-test-praktika)**

## 📊 Coursework — Recommendation Methods Comparison

**Dataset:** [VK-LSVD](https://huggingface.co/datasets/deepvk/VK-LSVD) — user–video
interactions from VKontakte. Subsample `up0.001_ip0.001`, weeks `00–02`.

**Compared models:**

| Model | Approach | Implementation |
|-------|----------|----------------|
| **MostPopular** | popularity baseline | pandas |
| **SVD** | matrix factorization | scikit-surprise |
| **ALS** | alternating least squares | scikit-surprise |
| **NCF** | neural collaborative filtering | PyTorch |

**Pipeline:**

- ⚖️ implicit feedback → a single rating: weighted actions (`like`, `share`, `bookmark`, …) + a log watch-time bonus;
- ⏱ **time-based split**: train on weeks 00–01, test on week 02 (no peeking into the future);
- 📏 metrics: **Precision@10, Recall@10, NDCG@10, MAP@10**;
- ⚡ faster inference: top-2000 candidates (SVD/ALS) and top-1000 (NCF), user subsampling.

### ▶️ Run

```bash
cd cursovaya26
pip install huggingface_hub pandas pyarrow numpy scikit-learn scikit-surprise torch

python download_data.py   # downloads weeks 00–02 into vk_data/ (if not there yet)
python train_models.py    # trains all models and prints metrics to the console
```

> 💡 If `scikit-surprise` fails to install, install `numpy<2` first.

## 🧪 Internship — Reksoft, "Databases"

The **CyberSportsPortal API Tests** collection in Bruno:

- 11 GET requests covering every endpoint of the test server;
- ≥ 2 assertions per request + JavaScript tests for response fields;
- a `baseUrl` environment, filtering via Query Params (`/api/Players/filter`).

Detailed README: [`prakt26/`](prakt26/) · standalone repo: [bruno-test-praktika](https://github.com/TWOINTHEPINKY/bruno-test-praktika)

## 📁 Repository structure

```text
ssu.vshtani26/
├── cursovaya26/
│   ├── download_data.py       # downloads VK-LSVD from Hugging Face
│   ├── train_models.py        # MostPopular / SVD / ALS / NCF + metrics
│   └── vk_data/               # weekly parquet files
├── prakt26/
│   ├── CyberSportsPortal API Tests/   # Bruno collection
│   └── README.md
└── README.md
```

---

👤 **Danila Fedotov** · SSU · [@TWOINTHEPINKY](https://github.com/TWOINTHEPINKY)
