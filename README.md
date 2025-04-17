# 🔐 Cybersecurity Attack Detection

Hackathon submission for **DSA4263 (Semester 2, 2025)**

This project focuses on detecting cybersecurity attacks using a combination of machine learning. Our data is sourced from well-established network intrusion detection datasets and enhanced with advanced preprocessing and graph metric features.

---

## 📁 Data Storage

All project datasets are stored in our [Google Drive Folder](https://drive.google.com/drive/folders/1IheCLjHlpWdNMZjccHZ7TaC7ZsoRBEwg?usp=sharing). Below is a summary of its contents:

### Folder Structure:

- **Raw Data**  
  Contains the original, unprocessed datasets from [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) and [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html).

- **Preprocessed Data**  
  Cleaned and merged datasets after passing PCAP files into CICFlowMeter, split into:
  - `all_2017.csv`
  - `ddos2018_cleaned.csv`

- **Train and Test Data for Baseline and AdaBoost Models**  
  Further processed data after feature selection and engineering, used to train and evaluate:
  - Logistic Regression (Baseline Model)
  - AdaBoost (Challenger Model)
  
  Split into: 
  - `traindata_2017_v1.csv`
  - `testdata_2018_v1.csv`


- **Train and Test Data Merged with Graph Features**  
  Enhanced datasets including graph-based metrics and used to train graph-feature-enhanced models: 
  - Logistic Regression with graph metrics (Challenger Model)
  - ADAboost with graph metrics (Challenger Model)
  
  Split into: 
  - `traindata_2017_v2.csv`
  - `testdata_2018_v2.csv`


> **Note**
> Train and test data for GAT (and other challenger models) were further processed from all_2017.csv and ddos2018_cleaned.csv. 
> Relevant Challenger Models: 
> - GAT (Selected model)  
> - MLP  
> - GCN  
> - TGAT  
> - Upgraded TGAT

---
## 📁 Repository Structure 

```
.
├── Models
│   ├── baseline_model.ipynb --> contains feature selection and engineering steps, Logistic Regression (Baseline Model) and Adaboost Model (Challenger Model)
│   ├── graph_ensemble_models.ipynb --> Models trained with added Graph Metrics (Challenger Models) and an Ensemble Model 
│   ├── gat.ipynb --> GAT (Challenger Model, final best performing model) 
│   ├── gat.py
│   ├── gcn.ipynb --> A3TGCN2 (Challenger Model)
│   ├── gcn.py
│   ├── mlp.ipynb --> Baseline Challenger Model (not focused on in report)
│   ├── mlp.py
│   ├── tgat.ipynb --> Challenger Model (not focused on in report)
│   ├── tgat.py
│   ├── upgraded_tgat.ipynb --> Challenger Model (not focused on in report)
│   └── upgraded_tgat.py
├── README.md
├── cybersec_EDA.ipynb --> contains Exploratory Data Analysis
├── data_preprocessing.ipynb --> contains data cleaning and merging steps
├── datadictionary.txt
└── requirements.txt

```

---

## 👩‍💻 Contributors

- Emma Lim Xiaoen, A0239300R
- Ang Zhiyong, A0233445J
- Chan Jin Xiang Filbert, A0245018L
- Jax Lee Le Sheng, A0236637X
- See Wee Shen Rachel, A0238731A

---

## 📌 License
This project is for academic use under the NUS DSA4263 course. Please do not use the data or results for commercial purposes.

