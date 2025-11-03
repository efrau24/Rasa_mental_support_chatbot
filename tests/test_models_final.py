import pandas as pd
import torch
import numpy as np
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
# Path modelli 
TEST_CSV_PATH = "./datasets/test_full.csv"
MODEL1_PATH = "./model1/best_model"
MODEL2_PATH = "./model2/best_model"

# =====================
# 1. Caricamento e inizializzazione modelli
# =====================

df = pd.read_csv(TEST_CSV_PATH)
tokenizer = RobertaTokenizer.from_pretrained(MODEL1_PATH)
model1 = RobertaForSequenceClassification.from_pretrained(MODEL1_PATH)
model2 = RobertaForSequenceClassification.from_pretrained(MODEL2_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device).eval()
model2.to(device).eval()



# =====================
# 2. Tokenizzazione 
# =====================

def tokenize_texts(texts):
    return tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
        )

# =====================
# 3. Predizioni con modello 1 (neutral vs non-neutral)
# =====================

df["label_model1"] = df["label"].apply(lambda x: 0 if x == "neutral" else 1)

inputs1 = tokenize_texts(df["text"].tolist()).to(device)
with torch.no_grad():
    outputs1 = model1(**inputs1)
    probs1 = F.softmax(outputs1.logits, dim=-1).cpu().numpy()
df["pred_model1"] = np.argmax(probs1, axis=1)
df["conf_model1_non_neutral"] = probs1[:, 1]

# # Salvataggio predizioni modello 1
# df.to_csv("predizioni_model1.csv", index=False)

# =====================
# 4. Filtraggio per modello 2
# =====================

filtered_df = df[
    (df["pred_model1"] == 1) &
    (df["conf_model1_non_neutral"] > 0.8) &
    (df["label"].isin(["change", "sustain"]))
].copy()
filtered_df["label_model2"] = filtered_df["label"].map({"change": 0, "sustain": 1})

# =====================
# 5. Predizioni modello 2
# =====================

inputs2 = tokenize_texts(filtered_df["text"].tolist()).to(device)
with torch.no_grad():
    outputs2 = model2(**inputs2)
    probs2 = F.softmax(outputs2.logits, dim=-1).cpu().numpy()
filtered_df["pred_model2"] = np.argmax(probs2, axis=1)

# # Salvataggio predizioni modello 2
# filtered_df.to_csv("predizioni_model2_confidence_filtered.csv", index=False)

# =====================
# 6. Calcolo metriche modello 1
# =====================

labels1 = df["label_model1"].values
preds1 = df["pred_model1"].values

precision1, recall1, f1_1, _ = precision_recall_fscore_support(labels1, preds1, average="weighted", zero_division=0)
acc1 = accuracy_score(labels1, preds1)

metrics_model1 = {
    "eval_accuracy": acc1,
    "eval_precision": precision1,
    "eval_recall": recall1,
    "eval_f1": f1_1
}

# =====================
# 7. Calcolo metriche modello 2
# =====================

labels2 = filtered_df["label_model2"].values
preds2 = filtered_df["pred_model2"].values

precision2, recall2, f1_2, _ = precision_recall_fscore_support(labels2, preds2, average="weighted", zero_division=0)
acc2 = accuracy_score(labels2, preds2)

metrics_model2 = {
    "eval_accuracy": acc2,
    "eval_precision": precision2,
    "eval_recall": recall2,
    "eval_f1": f1_2
}

# =====================
# 8. Salvataggio metriche 
# =====================

metriche_finali = {
    "model1": metrics_model1,
    "model2_filtered": metrics_model2,
}

with open("metriche.json", "w") as f:
    json.dump(metriche_finali, f, indent=4)


# =====================
# 9. Rappresentazione grafica delle metriche modello 1
# =====================

# Probabilità predette per la classe 'non-neutral'
y_probs_1 = df["conf_model1_non_neutral"].values
y_true_1 = df["label_model1"].values  

# Definizione soglie
thresholds_1 = np.linspace(0.5, 1.0, 101)
precisions_1, recalls_1, f1s_1 = [], [], []

# Calcolo metriche per ogni valore della soglia
for t in thresholds_1:
    y_pred_1 = (y_probs_1 >= t).astype(int)
    precisions_1.append(precision_score(y_true_1, y_pred_1, zero_division=0))
    recalls_1.append(recall_score(y_true_1, y_pred_1, zero_division=0))
    f1s_1.append(f1_score(y_true_1, y_pred_1, zero_division=0))

# Grafico
plt.figure(figsize=(10, 6))
plt.plot(thresholds_1, precisions_1, label='Precision', color='blue')
plt.plot(thresholds_1, recalls_1, label='Recall', color='green')
plt.plot(thresholds_1, f1s_1, label='F1 Score', color='red')
plt.xlabel('Soglia di confidenza (classe non-neutral)')
plt.ylabel('Valore metrica')
plt.title('Model 1: neutral vs non-neutral')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)



# =====================
# 10. Rappresentazione grafica delle metriche modello 2
# =====================

# Definizione soglie
thresholds_2 = np.linspace(0.5, 1.0, 101)
precisions_filt, recalls_filt, f1s_filt, counts = [], [], [], []

# Calcolo metriche per ogni valore della soglia
for t in thresholds_2:
    # Selezione frasi non-neutral con confidenza ≥ soglia
    sub_df = df[
        (df["pred_model1"] == 1) &
        (df["conf_model1_non_neutral"] >= t) &
        (df["label"].isin(["change", "sustain"]))
    ].copy()

    
    if sub_df.empty:
        # Nessun dato selezionato a questa soglia
        precisions_filt.append(0)
        recalls_filt.append(0)
        f1s_filt.append(0)
        counts.append(0)
        continue

    # Etichette vere
    sub_df["label_model2"] = sub_df["label"].map({"change": 0, "sustain": 1})

    # Predizioni modello 2
    inputs = tokenize_texts(sub_df["text"].tolist()).to(device)
    with torch.no_grad():
        outputs = model2(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

    preds = np.argmax(probs, axis=1)
    labels = sub_df["label_model2"].values

    # Metriche
    precisions_filt.append(precision_score(labels, preds, zero_division=0))
    recalls_filt.append(recall_score(labels, preds, zero_division=0))
    f1s_filt.append(f1_score(labels, preds, zero_division=0))
    counts.append(len(sub_df))

# Grafico
plt.figure(figsize=(10, 6))
plt.plot(thresholds_2, precisions_filt, label="Precision", color="blue")
plt.plot(thresholds_2, recalls_filt, label="Recall", color="green")
plt.plot(thresholds_2, f1s_filt, label="F1 Score", color="red")
plt.xlabel("Soglia di confidenza per filtro frasi non-neutral")
plt.ylabel("Valore metrica")
plt.title("Effetto della soglia di filtraggio su performance del Model 2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)



# =====================
# 11. conteggi predizioni non-neutral
# =====================

# Totale veri non-neutral (label originale diversa da "neutral")
true_non_neutral_total = (df["label_model1"] == 1).sum()
print(f"Totale veri non-neutral: {true_non_neutral_total}")

# Liste per salvare i valori
num_pred_non_neutral = []
num_correct_non_neutral = []

# Ciclo sulle soglie
for t in thresholds_1:
    # Predizioni come non-neutral a questa soglia
    mask_pred = (df["conf_model1_non_neutral"] >= t).astype(int)
    
    # Numero di predetti come non-neutral
    num_pred = mask_pred.sum()
    
    # Numero di veri non-neutral tra quelli predetti
    correct_pred = ((mask_pred == 1) & (df["label_model1"] == 1)).sum()
    
    num_pred_non_neutral.append(num_pred)
    num_correct_non_neutral.append(correct_pred)

# Grafico
plt.figure(figsize=(10, 6))
plt.plot(thresholds_1, num_pred_non_neutral, label="Predetti come non-neutral", color="orange")
plt.plot(thresholds_1, num_correct_non_neutral, label="Tra questi, veri non-neutral", color="green")
plt.axhline(y=true_non_neutral_total, color="red", linestyle="--", label="Totale veri non-neutral (costante)")
plt.xlabel("Soglia di confidenza (classe non-neutral)")
plt.ylabel("Numero di frasi")
plt.title("Conteggio predizioni non-neutral al variare della soglia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()