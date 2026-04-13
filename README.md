# PLMGraph-Ace

In this study, we propose **PLMGraph-Ace**, a deep learning framework for lysine acetylation site prediction.  
The method integrates a frozen protein language model with graph-based local dependency modeling. Specifically, it extracts contextual representations from ESM2 for 31-residue sequence windows centered on candidate lysine sites, and further constructs sequence-distance graphs to model local neighborhood relationships. The final prediction can be performed using a graph branch, an ESM-only branch, or a fusion branch.

1. Environment Setup
Firstly, create a virtual environment and install the required dependencies.
conda create -n plmgraphace python=3.10
conda activate plmgraphace
pip install -r requirements.txt

2. Pre-trained language model
This project uses the ESM2 model to extract contextual residue representations.
The pre-trained model parameters can be downloaded from the official ESM2 release or loaded from a local path.

3. Data preparation  
The dataset used in this project is stored in the `data` folder.  
Before running the script, please modify the corresponding data path in the code according to your local environment.

4. Running the model
The code for training is in train.py

Note：The first run may take longer because ESM2 representations need to be extracted and cached.
