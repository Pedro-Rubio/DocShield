import sys
sys.path.append('.')
from src.dataset.generator import generate_fraud_dataset
from src.dataset.labeler import apply_heuristics
import pandas as pd

print("Generando muestra de 20 registros...")
df = generate_fraud_dataset(n_legit=10, n_fraud=10)
df = apply_heuristics(df)

print("\n=== VISTA PREVIA DEL DATASET ===")
print("Total muestras: %d" % len(df))
print("Fraude: %d (%.1f%%)" % (df["is_fraud"].sum(), df["is_fraud"].mean()*100))
print("\nDistribución por tipo:")
print(df['fraud_type'].value_counts())

print("\n=== PRIMERAS 5 FILAS ===")
cols_to_show = ['blur_score', 'ela_score', 'ocr_confidence', 'moire_score', 'ip_risk_score', 'fraud_type', 'is_fraud']
print(df[cols_to_show].head().to_string())

print("\n=== ESTADÍSTICAS POR CLASE ===")
for cls in [0, 1]:
    label = 'FRAUDE' if cls == 1 else 'LEGIT'
    print("\n--- %s ---" % label)
    subset = df[df['is_fraud'] == cls]
    print(subset[['blur_score', 'ela_score', 'ocr_confidence', 'moire_score']].describe().round(2))

print("\n=== TIPOS DE FRAUDE ===")
fraud_only = df[df['is_fraud'] == 1]
print(fraud_only[['fraud_type', 'blur_score', 'ela_score', 'moire_score']].to_string())
