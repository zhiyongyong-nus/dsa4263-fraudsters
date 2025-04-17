import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from collections import defaultdict
import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

# =============== LOAD TRAINING DATA ===============
df = pd.read_csv('/content/drive/MyDrive/dsa4263/2017_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

df['Label'] = df['Label'].map({'BENIGN': 0}).fillna(1).astype(int)
df['MinuteBin'] = df['Timestamp'].dt.floor('Min')

graph_cols = ['Source IP', 'Destination IP', 'Label']
feature_cols = [c for c in df.columns if c not in graph_cols + ['Flow ID', 'Timestamp', 'MinuteBin']]

# Clean numeric features
df.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(imputer.fit_transform(df[feature_cols]))
df[feature_cols] = df[feature_cols].astype(np.float64)  # Enforce float64

# =============== NODE FEATURE ENGINEERING ===============
def build_node_features(sub_df):
    ip_stats = defaultdict(lambda: np.zeros(len(feature_cols) + 3))
    src_counts = sub_df['Source IP'].value_counts().to_dict()
    dst_counts = sub_df['Destination IP'].value_counts().to_dict()
    ip_dst_entropy = defaultdict(list)

    for _, row in sub_df.iterrows():
        src, dst = row['Source IP'], row['Destination IP']
        ip_stats[src][:len(feature_cols)] += row[feature_cols].values.astype(np.float64)
        ip_dst_entropy[src].append(dst)

    all_ips = set(sub_df['Source IP']).union(set(sub_df['Destination IP']))
    ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
    node_features = np.zeros((len(ip_to_idx), len(feature_cols) + 3))

    for ip, idx in ip_to_idx.items():
        feat = ip_stats[ip][:len(feature_cols)]
        count_src = src_counts.get(ip, 0)
        count_dst = dst_counts.get(ip, 0)
        ent = entropy(pd.Series(ip_dst_entropy[ip]).value_counts(normalize=True)) if ip_dst_entropy[ip] else 0
        node_features[idx] = np.concatenate([feat, [count_src, count_dst, ent]])

    return torch.tensor(node_features, dtype=torch.float32), ip_to_idx

# =============== GAT MODEL ===============
class EdgeGAT(torch.nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64):
        super().__init__()
        self.gat1 = GATConv(node_feat_dim, hidden_dim, heads=2, concat=True)
        self.gat2 = GATConv(2 * hidden_dim, hidden_dim, heads=1)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim + edge_feat_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x, edge_index, edge_attr, edge_src, edge_dst):
        x = torch.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        edge_inputs = torch.cat([x[edge_src], x[edge_dst], edge_attr], dim=1)
        return self.edge_mlp(edge_inputs)

# =============== TRAIN LOOP ===============
model = None
optimizer = None
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 5
minutes = df['MinuteBin'].unique()
print(f"Total 1-minute batches in training: {len(minutes)}")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}")
    epoch_loss = 0
    model_initialized = False

    for minute in tqdm(minutes):
        sub_df = df[df['MinuteBin'] == minute]
        if sub_df['Label'].nunique() < 2:
            continue

        node_feats, ip_to_idx = build_node_features(sub_df)
        edge_index = torch.tensor([
            [ip_to_idx[s] for s in sub_df['Source IP']],
            [ip_to_idx[d] for d in sub_df['Destination IP']]
        ], dtype=torch.long)

        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_attr = torch.tensor(sub_df[feature_cols].values.astype(np.float64), dtype=torch.float32)
        y = torch.tensor(sub_df['Label'].values, dtype=torch.long)

        if model is None:
            model = EdgeGAT(node_feat_dim=node_feats.shape[1], edge_feat_dim=edge_attr.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model_initialized = True

        model.train()
        optimizer.zero_grad()
        out = model(node_feats, edge_index, edge_attr, edge_src, edge_dst)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if not model_initialized:
        print("⚠️ No valid graphs in this epoch with both classes.")
        break

    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

# =============== EVALUATE ON TEST SET ===============
print("\n=== Loading and Evaluating Test Set (2018) ===")
test_df = pd.read_csv('/content/drive/MyDrive/dsa4263/ddos2018_cleaned.csv').iloc[:, :-2]
test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'], errors='coerce')
test_df = test_df.dropna(subset=['Timestamp'])

test_df['Label'] = test_df['Label'].map({'Benign': 0}).fillna(1).astype(int)
test_df['MinuteBin'] = test_df['Timestamp'].dt.floor('Min')

test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df[feature_cols] = scaler.transform(imputer.transform(test_df[feature_cols]))
test_df[feature_cols] = test_df[feature_cols].astype(np.float64)

test_minutes = test_df['MinuteBin'].unique()
all_preds = []
all_trues = []

print(f"🧪 Evaluating on {len(test_minutes)} 1-minute batches from test set...")

model.eval()
for minute in tqdm(test_minutes):
    sub_df = test_df[test_df['MinuteBin'] == minute]

    node_feats, ip_to_idx = build_node_features(sub_df)
    try:
        edge_index = torch.tensor([
            [ip_to_idx[s] for s in sub_df['Source IP']],
            [ip_to_idx[d] for d in sub_df['Destination IP']]
        ], dtype=torch.long)
    except KeyError:
        continue  # unseen IPs cannot be mapped

    edge_src = edge_index[0]
    edge_dst = edge_index[1]
    edge_attr = torch.tensor(sub_df[feature_cols].values.astype(np.float64), dtype=torch.float32)
    y_true = torch.tensor(sub_df['Label'].values, dtype=torch.long)

    with torch.no_grad():
        logits = model(node_feats, edge_index, edge_attr, edge_src, edge_dst)
        preds = torch.argmax(logits, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_trues.extend(y_true.cpu().numpy())

# =============== FINAL METRICS ===============
print("\n=== Final Test Set Evaluation ===")
print(classification_report(all_trues, all_preds, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_trues, all_preds))
