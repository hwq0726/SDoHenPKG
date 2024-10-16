# Leveraging Social Determinants of Health in Alzheimer’s Research Using LLM-Augmented Literature Mining and Knowledge Graphs
This repository holds the official code for the manuscript
"[Leveraging Social Determinants of Health in Alzheimer's Research Using LLM-Augmented Literature Mining and Knowledge Graphs](https://arxiv.org/abs/2410.09080)".
### 🦸‍ Abstract
Growing evidence suggests that social determinants of health (SDoH), a set of nonmedical factors, affect individuals' risks of developing Alzheimer's disease (AD) and related dementias. Nevertheless, the etiological mechanisms underlying such relationships remain largely unclear, mainly due to difficulties in collecting relevant information. This study presents a novel, automated framework that leverages recent advancements of large language model (LLM) and natural language processing techniques to mine SDoH knowledge from extensive literature and integrate it with AD-related biological entities extracted from the general-purpose knowledge graph PrimeKG. Utilizing graph neural networks, we performed link prediction tasks to evaluate the resultant SDoH-augmented knowledge graph. Our framework shows promise for enhancing knowledge discovery in AD and can be generalized to other SDoH-related research areas, offering a new tool for exploring the impact of social determinants on health outcomes.

### 📝 Requirements
To run the experiment, ensure you have the Deep Graph Library ([DGL](https://www.dgl.ai/)) installed. You can do so by running the following command:

```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

To deploy the environment, a recommended way is to use conda to create the environment and install the related packages shown as follows.

```bash
conda create -n SDoHenPKG python=3.10
pip install -r requirements.txt
conda activate SDoHenPKG 
```

### 🔨 Usage
All experiments are implemented on a SLURM cluster. You can find the relevant scripts in the [`slurm_scripts`](slurm_scripts) directory.
