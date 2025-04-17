import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

# ========== 1. LOAD DATA ==========
train = pd.read_csv('/content/drive/MyDrive/dsa4263/2017_data.csv')
test = pd.read_csv('/content/drive/MyDrive/dsa4263/ddos2018_cleaned.csv').iloc[:, :-2]

# ========== 2. CLEANING ==========
graph_cols = ['Source IP', 'Destination IP', 'Label']
feature_cols = [col for col in train.columns if col not in graph_cols + ['Flow ID', 'Timestamp']]

# Replace inf with NaN
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows that are fully empty or missing graph/feature columns
train.dropna(subset=graph_cols + feature_cols, inplace=True)
test.dropna(subset=graph_cols + feature_cols, inplace=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
train_imputed = imputer.fit_transform(train[feature_cols])
test_imputed = imputer.transform(test[feature_cols])

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(train_imputed)
X_test = scaler.transform(test_imputed)

# Encode labels
train['Label'] = train['Label'].map({'BENIGN': 0}).fillna(1).astype(int)
test['Label'] = test['Label'].map({'Benign': 0}).fillna(1).astype(int)

# ========== 3. BUILD GRAPH ==========
# Map IPs to node indices
all_ips = pd.concat([train['Source IP'], train['Destination IP'], test['Source IP'], test['Destination IP']]).unique()
ip_to_idx = {ip: idx for idx, ip in enumerate(all_ips)}
num_nodes = len(ip_to_idx)

# Node features (optional, 1-hot or dummy)
node_features = torch.eye(num_nodes)

def build_edge_index(df):
    return torch.tensor([
        [ip_to_idx[s] for s in df['Source IP']],
        [ip_to_idx[d] for d in df['Destination IP']]
    ], dtype=torch.long)

edge_index_train = build_edge_index(train)
edge_index_test = build_edge_index(test)

edge_attr_train = torch.tensor(X_train, dtype=torch.float32)
edge_attr_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(train['Label'].values, dtype=torch.long)
y_test = torch.tensor(test['Label'].values, dtype=torch.long)

# ========== 4. MODEL ==========
class EdgeGCN(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, edge_feat_dim):
        super().__init__()
        self.gcn1 = GCNConv(node_feat_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim + edge_feat_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x, edge_index, edge_attr, edge_src, edge_dst):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        edge_inputs = torch.cat([x[edge_src], x[edge_dst], edge_attr], dim=1)
        return self.edge_mlp(edge_inputs)

model = EdgeGCN(node_feat_dim=node_features.shape[1],
                hidden_dim=64,
                edge_feat_dim=edge_attr_train.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

# ========== 5. TRAIN ==========
edge_src = edge_index_train[0]
edge_dst = edge_index_train[1]

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(node_features, edge_index_train, edge_attr_train, edge_src, edge_dst)
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()
    if epoch%20 == 0:
      print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ========== 6. EVALUATE ==========
edge_src_test = edge_index_test[0]
edge_dst_test = edge_index_test[1]

model.eval()
with torch.no_grad():
    logits = model(node_features, edge_index_train, edge_attr_test, edge_src_test, edge_dst_test)
    preds = torch.argmax(logits, dim=1)

print("\n=== Evaluation ===")
print(classification_report(y_test.cpu(), preds.cpu(), digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test.cpu(), preds.cpu()))
