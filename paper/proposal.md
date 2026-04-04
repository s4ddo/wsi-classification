# **DL2 Proposal**
**Title:** Beyond Multiple Instance Learning: Extreme-Context Vision Transformers for Whole-Slide Image Classification
**Student:** [To be determined]
**Supervisor:** [Your Name]

### **1. Introduction & Motivation**
The prevailing paradigm for Whole-Slide Image (WSI) classification in computational pathology is Multiple Instance Learning (MIL) [1, 2]. Because a WSI digitized at 20x magnification frequently yields over 100,000 discrete patches, standard Vision Transformers (ViTs) encounter an intractable $\mathcal{O}(N^2)$ memory bottleneck in the self-attention mechanism. To circumvent this, MIL pipelines evaluate patches independently to extract features before applying a pooling heuristic (e.g., attention-weighted aggregation) to yield a slide-level prediction [1].

However, this two-stage methodology intrinsically relies on the Independent and Identically Distributed (IID) assumption, which is biologically flawed. The Tumor Microenvironment (TME) is highly structured; the spatial proximity and interaction between immune infiltrates, stromal tissue, and malignant cells carry critical prognostic value [3]. By severing these spatial priors, MIL struggles to capture long-range morphological dependencies. This project proposes abandoning MIL in favor of a native, single-sequence Vision Transformer. To resolve the extreme sequence length bottleneck, this research will primarily investigate advanced sparse attention mechanisms, with an optional exploration into distributed inference frameworks for massive scaling [4].

### **2. Project Objectives**
1. **Eliminate the MIL Aggregation Heuristic:** Formulate WSI classification as a unified, long-context sequence modeling task to natively preserve spatial and morphological context.
2. **Evaluate Attention Sparsity Paradigms:** Systematically contrast the representational capacity of **Deterministic Sparsity** (fixed spatial heuristics) against **Dynamic Sparsity** (learnable, content-routed attention) in retaining diagnostic signals.
3. **Optional Stretch Goal (Systems Scaling):** Investigate Distributed Sequence Parallelism, adapting frameworks like Distributed Sparse Attention (DSA) [4] to enable the processing of 100,000+ token sequences via KV-cache sharding across multiple GPUs.

### **3. Proposed Methodology & Theoretical Context**
The core architecture will adapt a ViT backbone to handle extreme context lengths, requiring the systematic evaluation of distinct sparsity paradigms:

* **Approach A: Deterministic (Spatial) Sparsity:** This approach relies on fixed topological priors to reduce algorithmic complexity to $\mathcal{O}(N \log N)$ or $\mathcal{O}(N)$. The student will implement sliding-window attention combined with dilated strides (analogous to Longformer [5]) to capture local cellular neighborhoods (e.g., tumor-stroma boundaries) while utilizing global `[CLS]` tokens to propagate slide-level summary data.
* **Approach B: Dynamic (Learnable) Sparsity:** To surpass the limitations of fixed windows, the student will implement content-aware routing mechanisms that dynamically construct the attention graph based on feature relevance [6].
    * *Top-K Routing & MoE:* Inspired by recent Large Language Model architectures (e.g., DeepSeek [6]), query tokens will generate routing scores to attend only to the Top-K most relevant keys globally. This allows distant but morphologically identical regions (e.g., primary tumor beds and distant micrometastases) to interact directly.
    * *Predictive Dropping:* Evaluating lightweight predictor networks to dynamically prune highly redundant tokens (e.g., vast background regions or uniform adipose tissue) prior to the self-attention computation [4].
* **Approach C: The Hybrid Pipeline & Optional Distributed Scaling:** The core deliverable for this phase is integrating deterministic local windows with dynamic global routing to create a unified Hybrid ViT. 
    * *Optional Extension:* To scale this hybrid model to its absolute limits without catastrophic Out-Of-Memory (OOM) failures, the student can explore the **DSA (Distributed Sparse Attention)** framework [4]. By applying sequence parallelism, the immense Key-Value (KV) cache can be sharded across multiple GPUs, drastically reducing peak VRAM requirements.

### **4. Evaluation Framework & Baselines**
The architectures will be benchmarked using complex gastrointestinal cohorts, specifically **TCGA-ESCA** (Esophageal Carcinoma) and **TCGA-STAD** (Stomach Adenocarcinoma), alongside the widely adopted **CAMELYON16** breast cancer metastasis dataset to ensure direct comparability with existing state-of-the-art MIL benchmarks.

Baselines are categorized to isolate the specific benefits of sequence modeling versus standard MIL:
* **Tier 1: Standard Attention-MIL:** AB-MIL [1] and CLAM [2] (to quantify the baseline penalty of the IID assumption).
* **Tier 2: Approximated Transformers:** TransMIL [7] (to evaluate whether native, exact sparse attention preserves more spatial information than TransMIL's Nyström-based mathematical approximation).
* **Tier 3: Proposed Extreme-Context ViT:** Ablation studies strictly comparing the Deterministic, Dynamic, and Hybrid sparsity models (with an optional assessment of latency/memory tradeoffs if DSA is implemented).

### **5. Success Metrics**
Validation will encompass both clinical efficacy and computational efficiency:
1.  **Clinical Metrics:** Area Under the Curve (AUC), F1-Score, and accuracy on slide-level classification.
2.  **Hardware & Efficiency Metrics:** * Peak VRAM consumption (GB) during forward-pass inference.
    * Inference throughput (latency per whole-slide).
    * *(If Approach C extension is attempted)* Maximum sequence scalability prior to OOM failure.

***

### **References**

[1] Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based deep multiple instance learning. *International Conference on Machine Learning (ICML)*, 2127-2136.
[2] Lu, M. Y., Williamson, D. F., Chen, T. Y., Chen, R. J., Barbieri, M., & Mahmood, F. (2021). Data-efficient and weakly supervised computational pathology on whole-slide images. *Nature Biomedical Engineering*, 5(6), 555-570.
[3] Chen, R. J., Chen, T. Y., Williamson, D. F., Shaban, M., Ferrara, S. J., ... & Mahmood, F. (2022). Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 16144-16155.
[4] Li, S., Lu, R., Chen, Q., Yin, H., Lyu, Y., Wen, Y., Tsang, I., & Zhang, T. (2026). DSA: Efficient Inference For Video Generation Models via Distributed Sparse Attention. *International Conference on Learning Representations (ICLR)*.
[5] Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*.
[6] DeepSeek-AI. (2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. *arXiv preprint arXiv:2405.04434*.
[7] Shao, Z., Bian, H., Chen, Y., Wang, Y., Zhang, J., Ji, X., ... & Zhang, Y. (2021). TransMIL: Transformer based correlated multiple instance learning for whole slide image classification. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 2136-2147.