# CAN Bus Intrusion Detection using LSTM Autoencoders

This project implements an **unsupervised anomaly detection** system for CAN bus traffic using a **PyTorch-based LSTM Autoencoder**. It is trained **only on attack-free data** to learn normal sequence patterns, and then used to detect anomalies in logs containing DoS or Fuzzy attacks.

---

## Overview
The CAN bus lacks built-in authentication, making it vulnerable to attacks like **message injection (DoS, Fuzzy)**.

This LSTM Autoencoder learns to reconstruct **normal temporal patterns**. Poor reconstruction indicates anomalies.

---

## Contents

- `parse.py`: Converts attack-free logs to a clean CSV format.
- `parse_attack.py`: Prepares raw DoS and Fuzzy attack CSVs for evaluation.

---

## Data Format

### Attack-Free CSV Format
```csv
timestamp,can_id,flags,dlc,data_0,data_1,data_2,data_3,data_4,data_5,data_6,data_7
```

### Parsed Attack Logs
- Format after `parse_attack.py`:
```csv
timestamp,can_id,flags,dlc,data_0,data_1,data_2,data_3,data_4,data_5,data_6,data_7,label
```
- `label` is **R (normal)** or **T (attack)**, used for evaluation only.

---

## Preprocessing & Features

### Feature Columns Used
- `time_delta` (calculated)
- `can_id`
- `dlc`
- `data_0` to `data_7`

### Steps
1. Compute `time_delta = df['timestamp'].diff().fillna(0)`
2. Use features in order.
3. Apply `MinMaxScaler` **only on training data**.
4. Apply same scaler to val/test data.

---

## Sequence Preparation
```python
SEQUENCE_LENGTH = 20

def create_sequences(data, L):
    return np.array([data[i:i+L] for i in range(len(data) - L + 1)])
```

Shape becomes: `(num_sequences, sequence_length, num_features)`

---

## Model Architecture

### LSTM Autoencoder
```python
LSTMAutoencoder(
  encoder: LSTM(input_size=11, hidden_size=64),
  decoder: LSTM(input_size=11, hidden_size=64),
  fc:     Linear(64 → 11)
)
```

- **Input**: Sequences of shape `(batch, seq_len, features)`
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- Early stopping on validation loss
- Best model checkpoint is saved and reloaded

---

## Threshold Selection

Compute reconstruction error on validation set:
```python
val_errors = model.predict(val_sequences)
```

Choose threshold using percentile:
```python
THRESHOLD = np.percentile(val_errors, 98)
```

---

## Evaluation

1. Load attack CSVs via `parse_attack.py`
2. Apply same preprocessing, scaler, and sequence builder
3. Predict errors:
```python
errors = model.predict(attack_sequences)
preds = (errors > THRESHOLD).astype(int)
```
4. Generate ground truth per sequence:
   - Use sliding window with OR rule over label column (1 if any T)
5. Compute:
   - Precision / Recall / F1 / Accuracy
   - Confusion Matrix
   - MSE distribution plots and thresholds

---

## Dependencies
```bash
pip install -r requirements.txt
```

## Dataset

**Hacking and Countermeasure Research Lab, Korea University**

https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset

---

## 📝 Citation

Please cite the dataset authors and credit this repo if used in academic work or development.
