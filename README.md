# Memory-Augmented Hierarchical Graph Networks for Interpretable Flow Cytometry Classification

<p align="center">
  <img src="Figures/MA-HGN_Architectural_Plot.pdf" alt="MA-HGN Framework Architecture" width="1000"/>
</p>
---

## 🔬 Overview

Flow cytometry generates high-dimensional single-cell measurements but requires expert manual analysis. Existing machine learning approaches treat cells independently, ignoring biological hierarchies and population-level interactions. 

**MA-HGN addresses this by:**
- 🧠 **Memory Bank**: Learning prototypical immune cell population signatures
- 🕸️ **Graph Neural Networks**: Modeling cell-cell phenotypic relationships via k-NN graphs
- 🏗️ **Hierarchical Aggregation**: Reasoning across 4 biological scales (cells → clusters → lineages → sample)
- ⚡ **Adaptive Fusion**: Automatically emphasizing the most diagnostic scale per sample

---

## 🎯 Key Features

- **State-of-the-Art Performance**: 90.5±3.8% mean accuracy across 3 datasets  
- **Cross-Disease Generalization**: Validates on COVID-19 (2 cohorts) and lupus nephritis  
- **Biomarker Discovery**: Independently identifies 6 clinically validated COVID-19 markers  
- **Interpretability**: Visualizes prototypes, hierarchical attention, and graph topology  
- **Computational Efficiency**: <300ms inference for 100,000 cells (near-linear complexity)  
- **Public Data**: Reproducible experiments on ImmPort datasets (SDY2011, SDY1708, SDY997)

---

## 🏗️ Architecture

**Input:** Variable-sized cell populations (N × 68 markers)  
**Output:** Binary disease classification + interpretable prototypes

**Pipeline:**
1. 🔄 **Set Transformer** → Permutation-invariant cell encoding  
2. 🧠 **Memory Bank** → 200 prototypes learn population signatures  
3. 🕸️ **Graph NN** → k-NN graphs capture cell-cell interactions  
4. 🏗️ **Hierarchy** → Aggregate across cells/clusters/lineages/sample  
5. ⚡ **Adaptive Fusion** → Attention-weighted combination → Classification

---

## 🧬 Discovered Biomarkers

MA-HGN independently identified 6 clinically validated COVID-19 markers through gradient-based feature importance:

| Marker | Rank | Biological Role | Clinical Validation |
|--------|------|----------------|---------------------|
| **CD_IgA** | #1 | Antibody response | Known COVID marker |
| **CD45** | #2 | Pan-leukocyte activation | Known COVID marker |
| **CD3** | #3 | T-cell identification | Known COVID marker |
| **CD11c** | #6 | Myeloid cell marker | Known COVID marker |
| **CD16** | #7 | NK cell/neutrophil marker | Known COVID marker |
| **CD45RA** | #9 | Naive vs. memory T-cells | Known COVID marker |

---

## 📁 Repository Structure
```bash
├── Configuration/
│   └── configuration.json           
├── Data Processing/
│   └── data_processing.py              
├── Evaluation Metrics/
│   └── evaluation_metrics.py   
├── Experiments/
│   ├── ablation.py
│   └── qualitative_experiments.py
├── Figures/
│   └── MA-HGN Architectural Plot.pdf
├── Model/
│   ├── loss.py
│   └── model.py                  
├── Visualization/
│   └── training_plots.py           
├── main.py
├── train.py
├── valid.py                                 
├── requirements.text                     
└── README.md                              


```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/dawoodrehman44/IEEE_EMBC_MA-HGN.git
cd IEEE_EMBC_2026

```
### Create environment
```bash
conda create -n MA-HGN_med python=3.8
conda activate MA-HGN_med

# Install dependencies
pip install -r requirements.txt
```

## Training
### Train the UAM_CXR Framework
```bash
python main.py \
    --model train \
    --config configuration/configuration.json \
    --data_path /path/to/training \

```

## Testing
### Perform Cell Embedding, Cluster Analysis, Prototypes_and_Marker_Analysis, Graph_Connection, and Hierarchy_Scale Analysis
```bash
python Experiments/qualitative_experiments.py \
    --Experiments/ablation.py \
    --data_path /path/to/validation \
```

## 🤝 Acknowledgments
We thank the creators of SDY2011, SDY997, SDY1708 datasets, and all the models used in this work, for making them publicly available to the community.

## Contact
For questions or collaborations, please contact: 
Dawood Rehman – [dawoodrehman1297@gapp.nthu.edu.tw]