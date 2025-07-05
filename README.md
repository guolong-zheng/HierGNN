# HierGNN: Hierarchical Graph Neural Network for Network Intrusion Detection

This repository contains the PyTorch implementation of HierGNN, a hierarchical graph neural network designed for network intrusion detection. The model leverages a two-tiered approach to analyze traffic data, capturing both fine-grained packet-level interactions and broader flow-level temporal dynamics.

## Model Architecture

HierGNN processes network traffic by modeling it at two distinct levels: the packet level and the flow level. This hierarchical structure allows the model to learn complex patterns that are often missed by traditional methods.

### Packet-Level Analysis (PacketLevelGAT)

- Each network flow is treated as a sequence of graphs, where each graph represents a short burst of packets.
- A Dynamic Feature Filter first selects the most salient features from raw packet data using an attention-based gating mechanism.
- A Graph Attention Network (GAT) is then applied to each packet graph to learn embeddings that capture the intricate relationships between packets.

### Flow-Level Temporal Analysis (TemporalModule)

- The sequence of packet-graph embeddings for a given flow is passed to a temporal GNN.
- Self-attention is used to weigh the importance of different packet-graphs within the flow.
- A bidirectional GRU processes the sequence of embeddings to capture temporal dependencies and the overall context of the entire flow.

### Generalization and Classification (GeneralizationModule)

- A final MLP-based module takes the aggregated flow embedding.
- It uses an adaptation mechanism to create a resilient feature representation before feeding it to a classifier for the final prediction (e.g., benign or malicious).

## Features

- **Hierarchical Structure**: Captures both micro (packet) and macro (flow) level traffic characteristics.
- **Dynamic Feature Selection**: Intelligently filters packet features to focus on the most relevant information.
- **Attention Mechanisms**: Utilizes attention at both the packet and flow levels for improved representation learning.
- **Temporal Modeling**: Employs a GRU to effectively learn from the sequence of network packets over time.
- **End-to-End Training**: The entire model is trained jointly, from raw packet features to final classification.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Geometric

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/HierGNN.git
   cd HierGNN
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   
   Install the requirements using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

The model expects preprocessed data stored in `.pt` files. The `hiergnn.py` script is designed to load these pre-made files.

### Data Format

Each sample in the `.pt` file should be a dictionary with the following keys:

- `'packet_graphs'`: A list of `torch_geometric.data.Data` objects.
- `'flow_features'`: A dictionary of flow-level features.
- `'labels'`: The ground-truth label.
- `'flow_metadata'`: A dictionary of flow metadata.

### Preprocessing

To use your own data, you will need to preprocess it into the format above. This typically involves:

1. Reading raw traffic data (e.g., from PCAP files).
2. Parsing network flows and extracting packets.
3. Generating graphs from packet interactions within a flow.
4. Extracting features for each packet.
5. Saving the processed data into `.pt` files.

## Usage

You can train and evaluate the HierGNN model by running the `hiergnn.py` script from your terminal.

### Command-Line Arguments

- `--dataset`: (Required) The name of the dataset to use. Choices: `cicids17`, `medbiot`.
- `--sample_ratio`: (Optional) The fraction of the dataset to use. Default: `1.0`.
- `--gpu_id`: (Optional) The ID of the GPU for training. Default: `3`.
- `--pre_sample`: (Optional) A flag to load pre-sampled data splits.

### Example

To train the model on the CIC-IDS2017 dataset using 10% of the data on GPU 0:

```bash
python hiergnn.py --dataset cicids17 --sample_ratio 0.1 --gpu_id 0
```

To run using pre-sampled data:

```bash
python hiergnn.py --dataset cicids17 --sample_ratio 0.1 --gpu_id 0 --pre_sample
```

## Code Structure

- `hiergnn.py`: Main script containing the complete model and training pipeline.
  - `HierGNN`: The top-level `nn.Module`.
  - `PacketLevelGAT`: The packet-level GAT model.
  - `TemporalGNN`: The flow-level GRU model.
  - `DynamicFeatureFilter`: The feature selection module.
  - `GeneralizationModule`: The final classification head.
  - `PreprocessedIoTDataset`: Custom PyTorch Dataset class.
  - `HierGNNTrainer`: Trainer class for handling training and evaluation.
  - `main()`: Main function to run the experiment.


## License

This project is open-source and available under the MIT License.
