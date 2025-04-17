import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from collections import defaultdict
import torch
import torch.nn as nn
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ---------------------------
# 0. Set device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. Load & Preprocess Training Data
# ---------------------------
df_train = pd.read_csv('/content/drive/MyDrive/dsa4263/2017_data.csv')
df_train['Timestamp'] = pd.to_datetime(df_train['Timestamp'], errors='coerce')
df_train = df_train.dropna(subset=['Timestamp'])

# Map label: benign flows become 0; everything else becomes 1
df_train['Label'] = df_train['Label'].map({'BENIGN': 0, 'Benign': 0}).fillna(1).astype(int)

# Define feature columns (exclude graph‐related columns)
graph_cols = ['Source IP', 'Destination IP', 'Label']
feature_cols = [col for col in df_train.columns if col not in graph_cols + ['Flow ID', 'Timestamp']]

# Clean numeric features
df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
df_train[feature_cols] = scaler.fit_transform(imputer.fit_transform(df_train[feature_cols]))
df_train[feature_cols] = df_train[feature_cols].astype(np.float64)

# ---------------------------
# 2. Enriched Node Feature Engineering
# ---------------------------
def build_node_features(sub_df):
    """
    For a given sub-dataframe (e.g. a window of events),
    aggregate the edge features per node and compute:
      - Number of times a node appears as source,
      - Number of times as destination,
      - Entropy over destination IPs for that node.
    The final node feature vector has dimension: len(feature_cols) + 3.
    """
    src_counts = sub_df['Source IP'].value_counts().to_dict()
    dst_counts = sub_df['Destination IP'].value_counts().to_dict()
    ip_dst_entropy = defaultdict(list)
    ip_feat_sum = defaultdict(lambda: np.zeros(len(feature_cols)))

    # Accumulate feature vectors per source IP and record destination occurrences.
    for _, row in sub_df.iterrows():
        src, dst = row['Source IP'], row['Destination IP']
        ip_feat_sum[src] += row[feature_cols].values.astype(np.float64)
        ip_dst_entropy[src].append(dst)

    # Get all IPs present (from sources or destinations)
    all_ips = set(sub_df['Source IP']).union(set(sub_df['Destination IP']))
    ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}

    # Build the node features: first the aggregated feature vector,
    # then append the counts (as source & destination) and entropy.
    node_features = np.zeros((len(ip_to_idx), len(feature_cols) + 3))
    for ip, idx in ip_to_idx.items():
        feat = ip_feat_sum[ip]
        count_src = src_counts.get(ip, 0)
        count_dst = dst_counts.get(ip, 0)
        ent = entropy(pd.Series(ip_dst_entropy[ip]).value_counts(normalize=True)) if ip_dst_entropy[ip] else 0
        node_features[idx] = np.concatenate([feat, [count_src, count_dst, ent]])
    return torch.tensor(node_features, dtype=torch.float32), ip_to_idx

# ---------------------------
# 3. Build a Dynamic Graph (Grouping by a Fixed Number of Events)
# ---------------------------
def preprocess_dynamic_by_count(df, feature_cols, window_size=200):
    """
    Create a dynamic graph signal from the DataFrame without minute binning.
    The data is divided into windows of consecutive events (default = 50 events per window).
    Each window is used to compute enriched node features, edge indices, edge attributes,
    and targets.
    """
    # Sort DataFrame by timestamp to preserve temporal order
    df = df.sort_values("Timestamp").reset_index(drop=True)

    num_windows = len(df) // window_size
    edge_indices = []
    edge_attrs = []
    targets = []
    features = []

    for i in range(num_windows):
        sub_df = df.iloc[i * window_size: (i + 1) * window_size]
        node_feats, ip_to_idx = build_node_features(sub_df)

        # Build edge index for the current window from source/destination IPs
        try:
            edge_index = torch.tensor([
                [ip_to_idx[s] for s in sub_df['Source IP']],
                [ip_to_idx[d] for d in sub_df['Destination IP']]
            ], dtype=torch.long)
        except KeyError:
            # Skip this window if a mapping issue occurs
            continue

        # Edge attributes are the original (scaled) features
        edge_attr = torch.tensor(sub_df[feature_cols].values.astype(np.float64), dtype=torch.float32)
        target = torch.tensor(sub_df['Label'].values, dtype=torch.long)

        edge_indices.append(edge_index)
        edge_attrs.append(edge_attr)
        targets.append(target)
        # The node features are expected in shape: (batch_size, num_nodes, in_channels, 1)
        features.append(node_feats.unsqueeze(0).unsqueeze(-1))

    return DynamicGraphTemporalSignal(edge_indices, edge_attrs, features, targets)

# Build the dynamic graph for training using fixed event grouping
train_dataset = preprocess_dynamic_by_count(df_train, feature_cols, window_size=50)

# ---------------------------
# 4. Define the A3TGCN2-based Edge Classifier
# ---------------------------
class A3TGCN2_EdgeClassifier(nn.Module):
    def __init__(self, in_channels, out_channels=2, batch_size=1):
        """
        in_channels should match the dimension of the enriched node features (len(feature_cols) + 3).
        """
        super().__init__()
        self.recurrent = A3TGCN2(in_channels=in_channels, out_channels=64, periods=1, batch_size=batch_size)
        self.edge_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index, edge_weight, edge_src, edge_dst):
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_src = edge_src.to(device)
        edge_dst = edge_dst.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
        # Get node representations from the recurrent module. Note we take the first element of the tuple.
        h = self.recurrent(x, edge_index, edge_weight)[0]
        h_src = h[edge_src]
        h_dst = h[edge_dst]
        # Concatenate the source and destination node features for each edge
        edge_input = torch.cat([h_src, h_dst], dim=1)
        return self.edge_mlp(edge_input)

# The enriched node features have dimension: len(feature_cols) + 3
in_channels = len(feature_cols) + 3
model = A3TGCN2_EdgeClassifier(in_channels=in_channels).to(device)

# ---------------------------
# 4.1 Initialize Optimizer, Scheduler, and Loss Function with Weighted Loss
# ---------------------------
# Here we use the support counts from your evaluation:
#   Support for class 0: 1,343,256
#   Support for class 1:   575,394
num_class0 = 1343256
num_class1 = 575394
total_samples = num_class0 + num_class1
weight_class0 = total_samples / (2.0 * num_class0)  # Inverse frequency weighting
weight_class1 = total_samples / (2.0 * num_class1)
weights = torch.tensor([weight_class0, weight_class1], dtype=torch.float).to(device)

# Use weighted CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss(weight=weights)

# Feel free to adjust the learning rate; here we use a lower LR based on earlier experimentation.
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ---------------------------
# 5. Training Loop
# ---------------------------
print("⏳ Training A3TGCN2 on 2017 data (grouped by fixed event count)...")
model.train()
num_epochs = 20 
for epoch in range(num_epochs):
    total_loss = 0.0
    # Iterate over each snapshot (each window of events)
    for t in range(len(train_dataset.features)):
        edge_index = train_dataset.edge_indices[t]
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        # Forward pass: note that we pass None for edge_weight
        y_hat = model(train_dataset.features[t], edge_index, None, edge_src, edge_dst)
        loss = loss_fn(y_hat, train_dataset.targets[t].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    scheduler.step()  # decay learning rate every `step_size` epochs
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}, LR: {current_lr:.6f}")

# ---------------------------
# 6. Preprocess & Evaluate on Test Set (2018)
# ---------------------------
df_test = pd.read_csv('/content/drive/MyDrive/dsa4263/ddos2018_cleaned.csv').iloc[:, :-2]
df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'], errors='coerce')
df_test = df_test.dropna(subset=['Timestamp'])

# Map label similarly for test (e.g. 'Benign' as 0, others as 1)
df_test['Label'] = df_test['Label'].map({'Benign': 0}).fillna(1).astype(int)
df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test[feature_cols] = scaler.transform(imputer.transform(df_test[feature_cols]))
df_test[feature_cols] = df_test[feature_cols].astype(np.float64)
test_dataset = preprocess_dynamic_by_count(df_test, feature_cols, window_size=50)

# Evaluation
model.eval()
all_preds, all_trues = [], []
print("\n🧪 Evaluating A3TGCN2 on 2018 test set (grouped by fixed event count)...")
with torch.no_grad():
    for t in range(len(test_dataset.features)):
        edge_index = test_dataset.edge_indices[t]
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        y_hat = model(test_dataset.features[t], edge_index, None, edge_src, edge_dst)
        preds = torch.argmax(y_hat, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(test_dataset.targets[t].cpu().numpy())

print("\n=== Final Test Set Evaluation ===")
print(classification_report(all_trues, all_preds, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_trues, all_preds))
