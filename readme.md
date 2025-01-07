# AMHP-SDC
Combination therapy is widely used to treat complex diseases, especially in patients who respond poorly to monotherapy. Identifying synergistic drug combinations (SDCs) poses a significant challenge due to the combinatorial complexity. Although graph representation learning algorithms have achieved success in predicting SDCs, predefined and rigid methods limit the capability of extracting complex semantic information, and high-order information within drug synergy data has not been adequately integrated with heterogeneous graph information. To address these challenges, we propose a novel framework, AMHP-SDC, which combines adaptive semantic meta-graph learning, hypergraph learning, and drug molecular graph perception for SDC prediction. The proposed method consists of four main components: (1) A Heterogeneous graph Information Extraction (HIE) module that automatically aggregates semantic information through adaptive semantic meta-graph searching; (2) A Biochemical Information Extraction (BIE) module that integrates hypergraph representation with GeniePath and MLP to generate node embeddings; (3) An Information Fusion module that combines multi-source information to produce a final representation; and (4) A Synergy Prediction module that constructs both primary and auxiliary tasks for SDC prediction. Computational experiment results demonstrate that AMHP-SDC outperforms the baseline methods on two benchmark datasets, confirming its effectiveness in predicting SDCs.

![image](overview-of-AMHP-SDC.jpg)

## Environment Requirement
Python == 3.11<br>
pandas == 2.2.2<br>
numpy == 1.26.3<br>
pytorch == 2.1.2<br>
scipy == 1.13.1<br>
scikit-learn == 1.5.1<br>
gensim == 4.3.3<br>

## Running the code
```sh
  # process data
  cd Model
  python preprocessed.py 
  # run model
  cd AMHPSDC
  python main.py
```
