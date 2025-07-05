import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset, Subset 
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch 
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import os
import time
from collections import Counter 
import sys
import argparse
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings('ignore')

class DynamicFeatureFilter(nn.Module):
    """Dynamic feature filtering module with attention-based gating"""

    def __init__(self, input_dim: int, hidden_dim: int, filter_ratio: float = 0.5):
        super(DynamicFeatureFilter, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.filter_ratio = filter_ratio

        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1), 
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        importance_scores = self.importance_net(x)

        if self.filter_ratio >= 1.0:
            mask = torch.ones_like(importance_scores) # No filtering
        else:
            threshold = torch.quantile(importance_scores, 1 - self.filter_ratio, dim=-1, keepdim=True)
            mask = (importance_scores >= threshold).float()

        gate_scores = self.gate(x)

        filtered_features = x * mask * gate_scores

        return filtered_features


class PacketLevelGAT(nn.Module):
    """Packet-level Graph Attention Network with dynamic filtering"""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(PacketLevelGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.feature_filter = DynamicFeatureFilter(input_dim, hidden_dim)

        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)

        self.node_attention_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1) 
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim) 

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_assignment: torch.Tensor) -> torch.Tensor:
        x_filtered = self.feature_filter(x)

        x = F.elu(self.gat1(x_filtered, edge_index))
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.layer_norm(x) 

        attention_scores = self.node_attention_scorer(x) 
        attention_weights = torch.softmax(attention_scores, dim=0) 

        graph_embeddings = global_mean_pool(x, batch_assignment) 

        return graph_embeddings


class TemporalGNN(nn.Module):
    """Temporal GNN with self-attention and GRU for flow-level analysis"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super(TemporalGNN, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.num_layers = num_layers

        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=4, 
            dropout=dropout,
            batch_first=True
        )

        
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.temporal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim) 
        )

        self.flow_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh()
        )

    def forward(self, packet_embeddings_padded: torch.Tensor, flow_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_len, embed_dim = packet_embeddings_padded.shape

        attn_mask = (torch.arange(max_len, device=packet_embeddings_padded.device).expand(batch_size, max_len) >= flow_lengths.unsqueeze(1))

        attended_features, _ = self.self_attention(
            packet_embeddings_padded, packet_embeddings_padded, packet_embeddings_padded,
            key_padding_mask=attn_mask
        )

        mask_for_temporal_features = (torch.arange(max_len, device=packet_embeddings_padded.device).expand(batch_size, max_len) < flow_lengths.unsqueeze(1))

        if flow_lengths.sum() > 0: 
            packed_input = nn.utils.rnn.pack_padded_sequence(
                attended_features, flow_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.gru(packed_input)
            gru_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=max_len
            )
        else:
            gru_output = torch.zeros(batch_size, max_len, self.hidden_dim * 2, device=packet_embeddings_padded.device)


        temporal_features = self.temporal_fusion(gru_output) 
        temporal_features = temporal_features * mask_for_temporal_features.unsqueeze(-1).float()

        flow_final_embeddings = []
        for i, length in enumerate(flow_lengths):
            if length > 0:
                valid_flow_features = temporal_features[i, :length]
                aggregated_flow_feature = self.flow_aggregator(valid_flow_features.mean(dim=0))
                flow_final_embeddings.append(aggregated_flow_feature)
            else:
                flow_final_embeddings.append(torch.zeros(self.hidden_dim, device=temporal_features.device))

        return torch.stack(flow_final_embeddings) 


class GeneralizationModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super(GeneralizationModule, self).__init__()

        self.mlp_adapt = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, f_agg: torch.Tensor) -> torch.Tensor:
        attention_scores = self.mlp_adapt(f_agg)
        a_adapt = F.softmax(attention_scores, dim=-1)
        f_resilient = f_agg * a_adapt

        logits = self.classifier(f_resilient)
        return logits


class HierGNN(nn.Module):
    def __init__(self,
                 packet_input_dim: int,
                 hidden_dim: int = 128,
                 num_classes: int = 2,
                 num_heads: int = 4, 
                 temporal_layers: int = 2, 
                 dropout: float = 0.1):
        super(HierGNN, self).__init__()

        self.packet_input_dim = packet_input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.packet_gnn = PacketLevelGAT(
            input_dim=packet_input_dim,
            hidden_dim=hidden_dim, 
            num_heads=num_heads,
            dropout=dropout
        )

        self.temporal_gnn = TemporalGNN(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim, 
            num_layers=temporal_layers,
            dropout=dropout
        )

        self.generalization_module = GeneralizationModule(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim, 
            num_classes=num_classes,
            dropout=dropout
        )


    def forward(self,
                batched_packet_graphs: Optional[Batch],
                num_packets_per_flow: torch.Tensor,
                flow_info: List[Dict] 
               ) -> torch.Tensor:
        batch_size = num_packets_per_flow.size(0)
        device = next(self.parameters()).device

        flow_packet_embeddings_list = []

        if batched_packet_graphs is not None and batched_packet_graphs.x is not None and batched_packet_graphs.x.numel() > 0:
            all_individual_packet_graph_embeddings = self.packet_gnn(
                batched_packet_graphs.x, 
                batched_packet_graphs.edge_index,
                batched_packet_graphs.batch 
            )
            current_idx = 0
            for num_packets in num_packets_per_flow:
                if num_packets > 0:
                    flow_embeddings = all_individual_packet_graph_embeddings[current_idx : current_idx + num_packets]
                    flow_packet_embeddings_list.append(flow_embeddings)
                else:
                    flow_packet_embeddings_list.append(torch.empty(0, self.hidden_dim, device=device))
                current_idx += num_packets.item() 
        else:
            for _ in range(batch_size):
                flow_packet_embeddings_list.append(torch.empty(0, self.hidden_dim, device=device))

        flow_lengths_for_temporal = torch.tensor([emb.size(0) for emb in flow_packet_embeddings_list], device=device, dtype=torch.long)

        max_flow_len_this_batch = 0
        if flow_packet_embeddings_list: 
             max_flow_len_this_batch = flow_lengths_for_temporal.max().item() if flow_lengths_for_temporal.numel() > 0 else 0

        effective_max_flow_len = max(1, max_flow_len_this_batch)

        padded_embeddings_for_temporal_gnn = []
        for emb in flow_packet_embeddings_list:
            num_packets_in_current_flow = emb.size(0)
            if num_packets_in_current_flow < effective_max_flow_len:
                padding_size = effective_max_flow_len - num_packets_in_current_flow
                padding = torch.zeros(padding_size, self.hidden_dim, device=device, dtype=emb.dtype) 
                emb_padded = torch.cat([emb, padding], dim=0)
            else: 
                emb_padded = emb
            padded_embeddings_for_temporal_gnn.append(emb_padded)

        if not padded_embeddings_for_temporal_gnn: 
            if batch_size > 0:
                stacked_embeddings_for_temporal_input = torch.zeros(batch_size, effective_max_flow_len, self.hidden_dim, device=device)
            else: 
                return torch.empty(0, self.num_classes, device=device) 
        else:
            stacked_embeddings_for_temporal_input = torch.stack(padded_embeddings_for_temporal_gnn)
            
        flow_contextual_features = self.temporal_gnn(stacked_embeddings_for_temporal_input, flow_lengths_for_temporal)
        
        logits = self.generalization_module(flow_contextual_features)
        
        return logits


class PreprocessedIoTDataset(Dataset):
    """Dataset class for loading preprocessed IoT data from .pt files"""

    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Preprocessed data file not found: {data_path}")

        print(f"Loading preprocessed data from {data_path}...")
        self.data_store = torch.load(data_path, map_location='cpu')

        required_keys = ['packet_graphs', 'flow_features', 'labels', 'flow_metadata']
        for key in required_keys:
            if key not in self.data_store:
                raise ValueError(f"Missing key '{key}' in preprocessed data in {data_path}")

        self.all_samples_packet_graphs = self.data_store['packet_graphs'] 
        self.all_samples_flow_features = self.data_store['flow_features'] 
        self.all_samples_labels = self.data_store['labels']           
        self.all_samples_flow_metadata = self.data_store['flow_metadata'] 

        print(f"Loaded {len(self.all_samples_labels)} samples from {data_path}")

    def __len__(self):
        return len(self.all_samples_labels)

    def __getitem__(self, idx):
        packet_graphs_for_sample = self.all_samples_packet_graphs[idx]

        processed_packet_graphs = []
        for graph_data in packet_graphs_for_sample:
            if not isinstance(graph_data, Data): 
                print(f"Warning: Item at index {idx} in packet_graphs is not a PyG Data object. Skipping graph.")
                continue

            if hasattr(graph_data, 'x') and graph_data.x is not None:
                graph_data.x = graph_data.x.float()
            if hasattr(graph_data, 'edge_index') and graph_data.edge_index is not None:
                graph_data.edge_index = graph_data.edge_index.long()

            processed_packet_graphs.append(graph_data.cpu())


        flow_features_for_sample = self.all_samples_flow_features[idx]
        label_for_sample = self.all_samples_labels[idx]
        flow_metadata_for_sample = self.all_samples_flow_metadata[idx]

        flow_info_combined = {}
        if isinstance(flow_features_for_sample, dict):
            flow_info_combined.update(flow_features_for_sample)
        if isinstance(flow_metadata_for_sample, dict):
            flow_info_combined.update(flow_metadata_for_sample)

        return {
            'packet_graphs': processed_packet_graphs, 
            'flow_info': flow_info_combined,
            'label': label_for_sample
        }

    def get_statistics(self):
        num_samples = len(self.all_samples_labels)
        label_counts = pd.Series(self.all_samples_labels).value_counts().to_dict()
        flow_lengths = [len(graphs) for graphs in self.all_samples_packet_graphs]
        avg_flow_length = np.mean(flow_lengths) if flow_lengths else 0
        min_flow_len = min(flow_lengths) if flow_lengths else 0
        max_flow_len = max(flow_lengths) if flow_lengths else 0

        return {
            'num_samples': num_samples,
            'label_distribution': label_counts,
            'avg_flow_length': avg_flow_length,
            'min_flow_length': min_flow_len,
            'max_flow_length': max_flow_len
        }


class HierGNNTrainer:
    def __init__(self, model: HierGNN, device: str = 'cpu', class_weights: Optional[torch.Tensor] = None):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.class_weights = class_weights.to(device) if class_weights is not None else None


    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, batch_data in enumerate(dataloader):
            optimizer.zero_grad()

            try:
                batched_packet_graphs_cpu = batch_data['batched_packet_graphs'] 
                num_packets_per_flow = batch_data['num_packets_per_flow'].to(self.device)
                flow_info_list = batch_data['flow_info'] 
                labels = batch_data['label'].to(self.device)

                batched_packet_graphs_gpu = None
                if batched_packet_graphs_cpu is not None:
                    batched_packet_graphs_gpu = batched_packet_graphs_cpu.to(self.device)
                logits = self.model(batched_packet_graphs_gpu, num_packets_per_flow, flow_info_list)

                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue 

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        return avg_loss, accuracy

    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float, Dict]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                try:
                    batched_packet_graphs_cpu = batch_data['batched_packet_graphs']
                    num_packets_per_flow = batch_data['num_packets_per_flow'].to(self.device)
                    flow_info_list = batch_data['flow_info']
                    labels = batch_data['label'].to(self.device)

                    batched_packet_graphs_gpu = None
                    if batched_packet_graphs_cpu is not None:
                        batched_packet_graphs_gpu = batched_packet_graphs_cpu.to(self.device)

                    logits = self.model(batched_packet_graphs_gpu, num_packets_per_flow, flow_info_list)
                    loss = criterion(logits, labels) 

                    total_loss += loss.item()

                    probabilities = F.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)

                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0, 'confusion_matrix': None}
        accuracy = 0.0
        if all_predictions and all_labels:
            accuracy = accuracy_score(all_labels, all_predictions)

            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)

            unique_labels = sorted(list(set(all_labels))) 
            precision_per_label, recall_per_label, f1_per_label, _ = \
                precision_recall_fscore_support(all_labels, all_predictions,
                                                labels=unique_labels, 
                                                average=None,
                                                zero_division=0)

            print("\n--- Per-Class Metrics ---")
            for i, label in enumerate(unique_labels):
                print(f"Label {label}: Precision={precision_per_label[i]:.4f}, Recall={recall_per_label[i]:.4f}, F1={f1_per_label[i]:.4f}")
            print("-------------------------\n")

            auc = 0.0
            if len(set(all_labels)) == 2 and len(all_probabilities) > 0:
                try:
                    probs_positive_class = [prob[1] for prob in all_probabilities]
                    auc = roc_auc_score(all_labels, probs_positive_class)
                except ValueError as e_auc: 
                    print(f"Could not compute AUC: {e_auc}")
                    auc = 0.0 

            cm = confusion_matrix(all_labels, all_predictions) 

            metrics = {
                'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc,
                'confusion_matrix': cm.tolist(), # Convert to list for easy printing/storage
                'per_class_precision': precision_per_label.tolist(),
                'per_class_recall': recall_per_label.tolist(),
                'per_class_f1': f1_per_label.tolist()
            }

        return avg_loss, accuracy, metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 1, lr: float = 0.001, weight_decay: float = 1e-5) -> Dict:
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True) # Schedule on val_loss

        best_val_f1 = -1.0 

        print("Starting training...")
        for epoch in range(num_epochs):
            start_time = time.perf_counter()

            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, val_metrics = self.validate(val_loader, criterion)

            scheduler.step(val_loss) 

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = self.model.state_dict().copy()
                print(f"Epoch {epoch:3d}: New best validation F1: {val_metrics['f1']:.4f}. Saving model.")

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            elapsed_time = time.perf_counter() - start_time
            print(f"Epoch {epoch:3d}: Time: {elapsed_time:.2f}s, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation F1: {best_val_f1:.4f}")

        final_val_loss, final_val_acc, final_val_metrics = self.validate(val_loader, criterion)
        print(f"Final validation metrics with best model: Acc: {final_val_acc:.4f}, F1: {final_val_metrics['f1']:.4f}, AUC: {final_val_metrics['auc']:.4f}")

        return {
            'best_val_f1': best_val_f1, # F1 of the epoch that was saved
            'final_val_metrics_with_best_model': final_val_metrics, # Metrics from running validate() again with the loaded best model
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }

    def evaluate_few_shot(self, test_loader: DataLoader) -> Dict:
        print("Evaluating performance with the loaded best model...")
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        test_loss, test_acc, test_metrics = self.validate(test_loader, criterion)

        print(f"Few-shot Test Results: Loss: {test_loss:.4f}")
        for metric_name, metric_value in test_metrics.items():
            if metric_name not in ['confusion_matrix', 'per_class_precision', 'per_class_recall', 'per_class_f1']:
                print(f"{metric_name.capitalize()}: {metric_value:.4f}")
            elif metric_name == 'confusion_matrix':
                print(f"Confusion Matrix:\n{np.array(metric_value)}")
        return test_metrics

    def plot_training_history(self):
        if not self.train_losses or not self.val_losses or not self.train_accuracies or not self.val_accuracies:
            print("No training history to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


def collate_fn_for_hier_gnn(batch_samples: List[Dict]) -> Dict:
    all_packet_graphs_for_batch = [] 
    num_packets_per_flow = []       
    for sample in batch_samples:
        flow_packet_graphs = sample['packet_graphs']

        valid_graphs_for_flow = [g for g in flow_packet_graphs if isinstance(g, Data) and g.x is not None and g.edge_index is not None]

        if valid_graphs_for_flow:
            all_packet_graphs_for_batch.extend(valid_graphs_for_flow)
            num_packets_per_flow.append(len(valid_graphs_for_flow))
        else:
            num_packets_per_flow.append(0) 

    batched_packet_graphs_obj = None
    if all_packet_graphs_for_batch:
        try:
            batched_packet_graphs_obj = Batch.from_data_list(all_packet_graphs_for_batch)
        except Exception as e:
            print(f"Error creating Batch from data list: {e}")
            batched_packet_graphs_obj = None

    return {
        'batched_packet_graphs': batched_packet_graphs_obj, 
        'num_packets_per_flow': torch.tensor(num_packets_per_flow, dtype=torch.long),
        'flow_info': [sample['flow_info'] for sample in batch_samples], 
        'label': torch.tensor([sample['label'] for sample in batch_samples], dtype=torch.long)
    }

def load_and_combine_data(dataset_name: str = "medbiot", sample_fraction: float = 0.1):

    if dataset_name == "cicids17":
        data_dir = "data/cicids2017/pt"
        train_files = ["Monday-WorkingHours_data.pt", "Tuesday-WorkingHours_data.pt", "Wednesday-workingHours_data.pt", "Thursday-WorkingHours_data.pt", "Friday-WorkingHours_data.pt"]
        train_paths = [os.path.join(data_dir, f) for f in train_files]
        train_datasets = [PreprocessedIoTDataset(path) for i, path in enumerate(train_paths)]

    elif dataset_name == "medbiot":
        data_dir = "data/medbiot/pt"
        train_file_map = {
            "bashlite_leg_data.pt": 0, "bashlite_mal_CC_all_data.pt": 0, "bashlite_mal_spread_all_data.pt": 0,
            "mirai_leg_data.pt": 1, "mirai_mal_CC_all_data.pt": 1, "mirai_mal_spread_all_data.pt": 1,
            "torii_leg_data.pt": 1, "torii_mal_all_data.pt": 0
        }
        train_datasets = [PreprocessedIoTDataset(os.path.join(data_dir, path)) for path, domain_id in train_file_map.items()]
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    combined_train_dataset = ConcatDataset(train_datasets)

    test_dataset = None
    train_val_pool = combined_train_dataset

    if sample_fraction < 1.0:
        print(f"\nOriginal combined dataset size: {len(combined_train_dataset)}")
        print(f"Performing stratified sampling for {sample_fraction * 100:.0f}% of the data...")

        def get_all_labels(ds):
            if isinstance(ds, PreprocessedIoTDataset):
                return ds.all_samples_labels
            elif isinstance(ds, ConcatDataset):
                return [label for sub_ds in ds.datasets for label in get_all_labels(sub_ds)]
            elif isinstance(ds, Subset):
                original_dataset = ds.dataset
                original_indices = ds.indices
                subset_labels = []
                all_original_labels = get_all_labels(original_dataset)
                for idx in original_indices:
                    subset_labels.append(all_original_labels[idx])
                return subset_labels
            return [ds[i]['label'] for i in range(len(ds))]

        all_labels = np.array(get_all_labels(combined_train_dataset))

        splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_fraction, random_state=42)

        dummy_x = np.zeros(len(all_labels))

        train_val_indices, test_indices = next(splitter.split(dummy_x, all_labels))

        test_dataset = Subset(combined_train_dataset, test_indices)
        train_val_pool = Subset(combined_train_dataset, train_val_indices)

    train_val_size = len(train_val_pool)
    val_size = int(train_val_size * 0.3)
    train_size = train_val_size - val_size
    train_dataset, val_dataset = random_split(train_val_pool, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset

def split_dataset(combined_dataset, val_split_ratio=0.3):
    total_size = len(combined_dataset)
    val_size = int(total_size * val_split_ratio)
    train_size = total_size - val_size

    if train_size <= 0 or val_size <= 0:
        if total_size == 0:
            raise ValueError("Combined dataset is empty.")
        elif total_size == 1:
            train_size = 1
            val_size = 0
            print("Warning: Dataset has only 1 sample. Assigning to training set only.")
        else: 
            train_size = total_size - 1
            val_size = 1
            print(f"Warning: Adjusted split for small dataset. Train: {train_size}, Val: {val_size}")

    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
    return train_dataset, val_dataset

def load_pre_sampled_data(dataset_name: str = "cicids17", sample_value: float = 0.1):
    train_val_pool = None
    test_dataset = None

    if dataset_name == "cicids17":
        data_dir = "data/cicids2017/sampled_pt"
        filename_part = ""
        if sample_value < 10:
            filename_part = f"{sample_value:.1f}pct"
        else:
            filename_part = f"{int(sample_value)}abs"


        train_val_path = os.path.join(data_dir, f"{dataset_name}_train_val_{filename_part}.pt")
        test_path = os.path.join(data_dir, f"{dataset_name}_test_{filename_part}.pt")

        try:
            train_val_pool = torch.load(train_val_path)
            test_dataset = torch.load(test_path)
            print("Successfully loaded pre-sampled datasets.")
        except FileNotFoundError as e:
            print(f"Error: A file was not found. Please check paths and filenames.")
            print(f"Details: {e}")
            exit(0)

    elif dataset_name == "medbiot":
        print(f"Loading logic for '{dataset_name}' is not yet implemented.")
        return None, None, None

    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    pool_size = len(train_val_pool)
    val_size = int(pool_size * 0.3)
    train_size = pool_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_val_pool, [train_size, val_size], generator=generator)

    return train_dataset, val_dataset, test_dataset


def load_preprocessed_data(data_dir: str = "preprocessed_iot_data"):
    train_path = os.path.join(data_dir, "train_data.pt")
    val_path = os.path.join(data_dir, "val_data.pt")
    test_path = os.path.join(data_dir, "test_data.pt")

    missing_files = []
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    if missing_files:
        raise FileNotFoundError(f"Missing preprocessed data files:\n" + "\n".join(missing_files) +
                                f"\nPlease run the preprocessing script first.")

    train_dataset = PreprocessedIoTDataset(train_path)
    val_dataset = PreprocessedIoTDataset(val_path)
    test_dataset = PreprocessedIoTDataset(test_path)

    return train_dataset, val_dataset, test_dataset

def calculate_class_weights(dataset: Dataset, device: torch.device) -> torch.Tensor:
    print("Calculating class weights...")
    labels = []

    def get_labels_from_dataset(ds):
        current_labels = []
        if isinstance(ds, PreprocessedIoTDataset):
            current_labels.extend(ds.all_samples_labels)
        elif isinstance(ds, Subset):
            for idx in ds.indices:
                current_labels.append(ds.dataset[idx]['label'])
        elif isinstance(ds, ConcatDataset):
            for sub_ds in ds.datasets:
                current_labels.extend(get_labels_from_dataset(sub_ds))
        else:
            print(f"Warning: Dataset of type {type(ds)} does not have 'all_samples_labels'. "
                  "Iterating through __getitem__ which might be slow for large datasets.")
            for i in range(len(ds)):
                current_labels.append(ds[i]['label'])
        return current_labels

    labels.extend(get_labels_from_dataset(dataset))

    if not labels:
        print("No labels found in dataset for class weight calculation. Returning None.")
        return None

    label_counts = Counter(labels)
    num_classes = len(label_counts)

    if num_classes < 2:
        print(f"Only {num_classes} class found. Class weighting not applicable. Returning None.")
        return None

    sorted_labels = sorted(label_counts.keys())

    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    max_count = max(label_counts.values())

    for label in sorted_labels:
        count = label_counts[label]
        weight = float(max_count / count) 
        class_weights[label] = weight

    print(f"Calculated class weights: {class_weights.tolist()}")
    return class_weights.to(device)


def main(dataset: str, sample_ratio: float, gpu_id: int, pre_sample: bool):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    try:
        if pre_sample:
            train_dataset_raw, val_dataset_raw, test_dataset_raw = load_pre_sampled_data(dataset, sample_ratio)
        else:
            train_dataset_raw, val_dataset_raw, test_dataset_raw = load_and_combine_data(dataset, sample_ratio)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None

    class_weights = calculate_class_weights(train_dataset_raw, device)

    print(f"\nDataset Loaded")

    
    batch_s = 128 
    num_w = 2 
    train_loader = DataLoader(train_dataset_raw, batch_size=batch_s, shuffle=True, collate_fn=collate_fn_for_hier_gnn, num_workers=num_w, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset_raw, batch_size=batch_s, shuffle=False, collate_fn=collate_fn_for_hier_gnn, num_workers=num_w, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset_raw, batch_size=batch_s, shuffle=False, collate_fn=collate_fn_for_hier_gnn, num_workers=num_w, pin_memory=True if device.type == 'cuda' else False)

    packet_feat_dim = 10 
    hid_dim = 128       
    num_cls = 2        
    gat_heads = 8      
    temp_layers = 3     
    drop_rate = 0.2     

    model = HierGNN(
        packet_input_dim=packet_feat_dim,
        hidden_dim=hid_dim,
        num_classes=num_cls,
        num_heads=gat_heads,
        temporal_layers=temp_layers,
        dropout=drop_rate
    )

    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    trainer = HierGNNTrainer(model, device=str(device), class_weights=class_weights)

    num_eps = 20 
    learn_rate = 0.0005 
    wd = 1e-5

    print("\nStarting training...")
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_eps,
        lr=learn_rate,
        weight_decay=wd
    )

    print(f"\nTraining completed. Best validation F1: {training_results['best_val_f1']:.4f}")

    print("\nEvaluating few-shot performance on the test set...")
    few_shot_results = trainer.evaluate_few_shot(test_loader)

    trainer.plot_training_history()

    return model, training_results, few_shot_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment with a selected dataset.")

    parser.add_argument('--dataset', type=str, required=True, choices=['cicids17', 'medbiot'])
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--pre_sample', action='store_true')

    args = parser.parse_args()

    model_trained, results_training, results_fewshot = main(args.dataset, args.sample_ratio, args.gpu_id, args.pre_sample)

    if model_trained is not None and results_fewshot is not None:
        print("\nExperiment completed successfully!")
        print(f"Final test accuracy with best model: {results_fewshot.get('accuracy', 'N/A'):.4f}")
    else:
        print("\nExperiment failed or did not run. Please check logs and data paths.")