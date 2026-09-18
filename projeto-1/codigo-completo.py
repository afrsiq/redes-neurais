# Transparência sobre o uso de IA: Código gerado pelo Gemini

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

df = pd.read_csv('C:/Users/anafl/Downloads/dataset_projeto1.csv')
X = df[['x']].values
y = df[['y']].values

# Divisão: 10% Treino, 10% Validação, 80% Teste
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.80, random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

# Normalização Z-Score (calculada estritamente sobre o conjunto de Treino para evitar Data Leakage)
mean_X, std_X = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
mean_y, std_y = y_train.mean(axis=0), y_train.std(axis=0) + 1e-8

X_train_norm = (X_train - mean_X) / std_X
X_val_norm   = (X_val - mean_X) / std_X
X_test_norm  = (X_test - mean_X) / std_X

y_train_norm = (y_train - mean_y) / std_y
y_val_norm   = (y_val - mean_y) / std_y
y_test_norm  = (y_test - mean_y) / std_y

print("--- Primeiras Linhas ---")
print(df.head())

print("\n--- Resumo Estatístico ---")
print(df.describe())

# 2. Plotagem gráfica (Scatter Plot para dados 2D)
plt.figure(figsize=(8, 4))
plt.scatter(df['x'], df['y'], color='#1f77b4', alpha=0.7, label='Amostras')
plt.title('Distribuição dos Dados (x vs y)')
plt.xlabel('Variável de Entrada (x)')
plt.ylabel('Alvo (y)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

train_dataset = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(y_train_norm, dtype=torch.float32))
val_dataset   = TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32), torch.tensor(y_val_norm, dtype=torch.float32))
test_dataset  = TensorDataset(torch.tensor(X_test_norm, dtype=torch.float32), torch.tensor(y_test_norm, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, dropout_prob=0.0):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
def train_model(use_l1=False, l1_lambda=0.005, use_l2=False, l2_lambda=0.01,
                dropout_prob=0.0, use_momentum=False, momentum_val=0.9, lr=0.01, epochs=300):

    torch.manual_seed(42)
    model = MLP(dropout_prob=dropout_prob)
    criterion = nn.MSELoss()

    weight_decay = l2_lambda if use_l2 else 0.0
    momentum = momentum_val if use_momentum else 0.0
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Penalização Regularização L1 (Lasso)
            if use_l1:
                l1_penalty = sum(torch.sum(torch.abs(p)) for p in model.parameters())
                loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # Validação
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val_norm, dtype=torch.float32)
            val_y = torch.tensor(y_val_norm, dtype=torch.float32)
            val_outputs = model(val_x)
            v_loss = criterion(val_outputs, val_y).item()
            val_losses.append(v_loss)

    return model, train_losses, val_losses

experiments = {
    "Baseline": {"use_l1": False, "use_l2": False, "dropout_prob": 0.0, "use_momentum": False},
    "L1": {"use_l1": True, "l1_lambda": 0.005, "use_l2": False, "dropout_prob": 0.0, "use_momentum": False},
    "L2": {"use_l1": False, "use_l2": True, "l2_lambda": 0.01, "dropout_prob": 0.0, "use_momentum": False},
    "Dropout": {"use_l1": False, "use_l2": False, "dropout_prob": 0.1, "use_momentum": False},
    "Momentum": {"use_l1": False, "use_l2": False, "dropout_prob": 0.0, "use_momentum": True, "momentum_val": 0.9},
    "L2 + Dropout": {"use_l1": False, "use_l2": True, "l2_lambda": 0.01, "dropout_prob": 0.1, "use_momentum": False},
    "L2 + Momentum": {"use_l1": False, "use_l2": True, "l2_lambda": 0.01, "dropout_prob": 0.0, "use_momentum": True, "momentum_val": 0.9},
    "Dropout + Momentum": {"use_l1": False, "use_l2": False, "dropout_prob": 0.1, "use_momentum": True, "momentum_val": 0.9},
}

results = {}
metrics_list = []
test_x_tensor = torch.tensor(X_test_norm, dtype=torch.float32)

for name, kwargs in experiments.items():
    model, t_loss, v_loss = train_model(**kwargs)

    # Avaliação no conjunto de teste
    model.eval()
    with torch.no_grad():
        preds_norm = model(test_x_tensor).numpy()

    # Despadronização Z-Score das predições para a escala original de Y
    preds = preds_norm * std_y + mean_y

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results[name] = {"train_loss": t_loss, "val_loss": v_loss, "preds": preds}
    metrics_list.append({
        "Modelo / Configuração": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2
    })
df_metrics = pd.DataFrame(metrics_list)
print(df_metrics.to_string(index=False))

# Gráfico da Convergência do Baseline (Erro de Treino e Validação) e Gráfico de Convergência dos Modelos
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()

for i, (name, res) in enumerate(results.items()):
    ax = axes[i]
    ax.plot(res["train_loss"], label="Treino", color="#1f77b4")
    ax.plot(res["val_loss"], label="Validação", color="#d62728", linestyle="--")
    ax.set_title(f"Convergência: {name}")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Perda (MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("convergencia_modelos.png")
plt.show()

preds_base = results["Baseline"]["preds"]
preds_best = results["Momentum"]["preds"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(y_test, preds_base, alpha=0.5, color='gray', label='Baseline (SGD)')
ax1.scatter(y_test, preds_best, alpha=0.7, color='blue', label='Melhor Modelo (Momentum)')
min_val = min(y_test.min(), preds_base.min(), preds_best.min())
max_val = max(y_test.max(), preds_base.max(), preds_best.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal ($y = \\hat{y}$)')
ax1.set_title('Parity Plot: Valor Real vs. Predição (Teste)')
ax1.set_xlabel('Valor Real ($y$)')
ax1.set_ylabel('Valor Predito ($\\hat{y}$)')
ax1.legend()
ax1.grid(True, alpha=0.3)

residuos = y_test - preds_best
ax2.scatter(preds_best, residuos, alpha=0.7, color='green')
ax2.axhline(0, color='red', linestyle='--', label='Resíduo Zero')
ax2.set_title('Gráfico de Resíduos (Melhor Modelo - Momentum)')
ax2.set_xlabel('Valores Preditos ($\\hat{y}$)')
ax2.set_ylabel('Resíduos ($y - \\hat{y}$)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("parity_residuos.png")
plt.show()
