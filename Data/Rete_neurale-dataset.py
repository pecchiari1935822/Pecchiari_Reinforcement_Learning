
# Cell 1 - import e configurazione
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Cell 2 - importa il preprocessing già eseguito
"""Importiamo il modulo Preparazione_dataset.py che contiene i dati già puliti e normalizzati. Dividiamo in:
- X_train, y_train: dati di addestramento (70% circa)
- X_val, y_val: dati di validazione (durante training, monitora il modello)
- X_test, y_test: dati di test (valutazione finale, mai visti dal modello)
- scaler_X, scaler_y: oggetti che convertono i dati tra scala normalizzata e originale
Perché normalizzare? Reti neurali convergono più velocemente quando i dati sono in range [0,1] o [-1,1]. 
Lo scaler ci permette poi di convertire le predizioni alla scala originale per interpretarle."""

import Data.Preparazione_dataset as prep

# Ottieni i dati scalati e gli splitter (usiamo i dati puliti e normalizzati dopo rimozione outlier)
#Dal file di pulizia del dtaset viengono presi i dof e gli of già normalizzati e divisi
X_train = prep.X_scaled_compl
y_train = prep.y_scaled_compl
X_val   = prep.X_val_scaled_compl
y_val   = prep.y_val_scaled_compl
X_test  = prep.X_test_scaled_compl
y_test  = prep.y_test_scaled_compl

# Scaler per riportare le previsioni alla scala originale (vengono presi gli stessi scaler del file preparazione dataset)
scaler_y = prep.scaler_y
scaler_X = prep.scaler_X

print("\nShapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)


# Cell 3 - definizione del modello (MLP per regressione)
"""Input (128 feature) 
   ↓
   Dense 128 neuroni + ReLU
   ↓
   Dropout 20%
   ↓
   Dense 64 neuroni + ReLU
   ↓
   Dense 32 neuroni + ReLU
   ↓
   Output (neuroni di output)"""

"""Scelte tecniche:
- Dense 128 → 64 → 32: scaliamo gradualmente per imparare rappresentazioni sempre più astratte (pattern complessi → pattern semplici)
- ReLU (Rectified Linear Unit): activation function che introduce non-linearità, permette al modello di imparare relazioni complesse
- Dropout 20%: disattiva casualmente il 20% dei neuroni durante training per evitare overfitting (il modello memorizza i dati invece di generalizzare)
- Output lineare: per regressione (non vogliamo vincolare l'output a un range specifico)
- Optimizer Adam: adattamento automatico del learning rate, converge rapidamente
- Loss MSE (Mean Squared Error): penalizza gli errori grandi, standard per regressione"""

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

def build_model(input_dim, output_dim, lr=1e-3):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),             #layer di input con dimensione dei dati di addestramento
        layers.Dense(64, activation='relu'),   #2° (hydden) layer con 128 neuroni e funzione di attivazione ReLU
        layers.Dropout(0.2),                          #dropout del 20% (spegnimento del 20% dei neuroni)per regolarizzazione
        layers.Dense(32, activation='relu'),    #3° (hydden) layer con 64 neuroni e funzione di attivazione ReLU
            #4° (hydden) layer con 32 neuroni e funzione di attivazione ReLU
        layers.Dense(output_dim, activation='linear') #layer di output con numero di neuroni pari alla dimensione dell'output e attivazione lineare (per regressione)
    ])
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

model = build_model(input_dim, output_dim)   #creazione della rete neurale (utilizzo della funzione build_model con i parametri di input e output)
model.summary()  #stampa la struttura del modello (numero di layer, neuroni, parametri)


# Cell 4 - callbacks e cartelle
"""I callback sono "monitor" che controllano il training:
- EarlyStopping: se la validazione loss non migliora per 25 epoche consecutive, interrompe il training (evita overfitting)
- ModelCheckpoint: salva automaticamente il modello migliore (quello con la validazione loss più bassa)
Questo previene di addestrare troppo e "rovinare" il modello."""

os.makedirs('models', exist_ok=True)
checkpoint_path = os.path.join('models', 'best_model.keras')  # Cambia da .h5 a .keras

es = callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
mc = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

# Cell 5 - training
"""Addestriamo il modello per massimo 500 epoche:
- batch_size=32: aggiorniamo i pesi ogni 32 campioni (compromesso tra velocità e stabilità)
- validation_data=(X_val, y_val): ad ogni epoca, valutiamo su dati non visti per monitorare overfitting
- verbose=2: mostriamo il progresso
Cosa succede: il modello vede i dati di training, calcola l'errore, aggiorna i pesi, ripete finché non converge."""

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=32,      # Indica quanti esmpi alla volta analizza in ogni epoca per aggiornare i pesi (alla fine in un epoca viene visto tutto il dataset)
    callbacks=[es, mc], # Aggiunge i callback per 'early stopping' e 'model checkpoint'
    verbose=2           # quante righe vengono stampate a schermo durante il training (0 = niente, 1 = barra di progresso, 2 = una riga per epoca
)

# Cell 6 - plot delle metriche di training
"""tracciamo due grafici:
- Loss (MSE): errore medio. Se il gap tra train e validation cresce = overfitting
- MAE (Mean Absolute Error): errore medio assoluto, più interpretabile"""

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend(); plt.title('Loss (MSE)')
plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='train mae')
plt.plot(history.history['val_mae'], label='val mae')
plt.legend(); plt.title('MAE')
plt.tight_layout()
plt.show()

# Cell 7 - valutazione su test set (scalato)
"""Misuriamo le prestazioni su dati mai visti durante training. 
Se il test loss è molto più alto della validation loss = il modello non generalizza bene."""

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Mean Square Error 'MSE' (scaled): {test_loss:.6f}, \nTest Mean absolute error 'MAE' (scaled): {test_mae:.6f}")

val_loss_finale = history.history['val_loss'][-1]  # ultimo valore di validazione
print('\nSe il test loss è molto più alto della validation loss = il modello non generalizza bene')
print(f"Validation loss (finale): {val_loss_finale:.6f}")
print(f"Test loss: {test_loss:.6f}")
differenza = (test_loss - val_loss_finale)*100
print(f"Differenza tra i due: {differenza:.2f}%")

# Cell 8 - predizioni e riportare alla scala originale
"""Facciamo predizioni su X_test
Usiamo scaler_y.inverse_transform() per convertire dalla scala normalizzata alla scala originale (altrimenti i numeri non sono interpretabili)
- RMSE (Root Mean Squared Error): radice dell'errore quadratico medio. Se RMSE=10 e il dato varia da 0 a 100, l'errore è del 10%
- R² (Coefficient of determination): quanto il modello spiega la variabilità dei dati. R²=0.9 significa il modello spiega il 90% della varianza"""

"""Il coefficiente R² è il più importante:
R² > 0.9: Eccellente, il modello spiega il 90%+ della varianza ✓✓✓
R² = 0.7 - 0.9: Buono, spiega il 70-90% della varianza ✓✓
R² = 0.5 - 0.7: Accettabile, ma potrebbe migliorare ✓
R² < 0.5: Scadente, il modello non apprende bene ✗
R² < 0: Molto cattivo, il modello è peggiore di una media costante ✗✗"""

y_pred_scaled = model.predict(X_test)               #predico l'output dell'input di test
y_pred = scaler_y.inverse_transform(y_pred_scaled)  #levo la normallizzazione (lo scalare) all'utput appena predetto
y_test_orig = scaler_y.inverse_transform(y_test)    #mi segno l'output vero corrispondente all'input di cui ho predetto l'output

# Metriche in scala originale
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)
print(f"\nTest RMSE (original scale): {rmse:.6f}")
print(f"Test R2: {r2:.6f}")

data_range = y_test_orig.max() - y_test_orig.min()
rmse_percentage = (rmse / data_range) * 100
print(f"RMSE come % del range: {rmse_percentage:.2f}%")

# Calcola metriche dettagliate per ogni output su TUTTI i campioni
print("\n=== ERRORI DETTAGLIATI PER OGNI OUTPUT ===\n")

metriche_per_output = []

for i in range(y_test_orig.shape[1]):
    mse_i = mean_squared_error(y_test_orig[:, i], y_pred[:, i])
    rmse_i = np.sqrt(mse_i)
    r2_i = r2_score(y_test_orig[:, i], y_pred[:, i])
    mae_i = np.mean(np.abs(y_test_orig[:, i] - y_pred[:, i]))

    # Errore percentuale rispetto al range dell'output
    range_i = y_test_orig[:, i].max() - y_test_orig[:, i].min()
    rmse_perc_i = (rmse_i / range_i * 100) if range_i != 0 else 0

    # MAPE: evita divisione per zero
    mask = y_test_orig[:, i] != 0
    mape_i = np.mean(
        np.abs((y_test_orig[:, i][mask] - y_pred[:, i][mask]) / y_test_orig[:, i][mask])) if mask.any() else 0

    metriche_per_output.append({
        'Output': i,
        'MSE': mse_i,
        'RMSE': rmse_i,
        'RMSE %': rmse_perc_i,
        'MAE': mae_i,
        'MAPE': mape_i,
        'R²': r2_i
    })

    print(f"Output {i:2d}: MSE={mse_i:.6f} | RMSE={rmse_i:.6f} ({rmse_perc_i:5.2f}%) | MAE={mae_i:.6f} | MAPE={mape_i:.6f} | R²={r2_i:.6f}")

n_outputs = y_test_orig.shape[1]
labels = [f'Output {m["Output"]}' for m in metriche_per_output]
rmse_vals = [m['RMSE'] for m in metriche_per_output]
mae_vals  = [m['MAE']  for m in metriche_per_output]
mape_vals = [m['MAPE'] for m in metriche_per_output]
r2_vals   = [m['R²']   for m in metriche_per_output]

x = np.arange(n_outputs)
width = 0.3

# ── GRAFICO 1: Barre delle metriche ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Metriche per output', fontsize=14, fontweight='bold')

for ax, vals, title, color in zip(
    axes,
    [rmse_vals, mae_vals, mape_vals],
    ['RMSE', 'MAE', 'MAPE'],
    ['steelblue', 'coral', 'mediumseagreen']
):
    ax.bar(x, vals, color=color, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.grid(axis='y', linestyle='--', alpha=0.5)


plt.tight_layout()
plt.show()


# Crea un DataFrame con tutte le metriche
df_metriche = pd.DataFrame(metriche_per_output)


# Statistiche generali
print(f"\n--- STATISTICHE GLOBALI ---")
print(f"R² medio: {df_metriche['R²'].mean():.6f}")
print(f"R² peggiore: {df_metriche['R²'].min():.6f} (Output {df_metriche['R²'].idxmin()})")
print(f"R² migliore: {df_metriche['R²'].max():.6f} (Output {df_metriche['R²'].idxmax()})")
print(f"RMSE % medio: {df_metriche['RMSE %'].mean():.2f}%")
print(f"RMSE % peggiore: {df_metriche['RMSE %'].max():.2f}% (Output {df_metriche['RMSE %'].idxmax()})")
print(f"RMSE % migliore: {df_metriche['RMSE %'].min():.2f}% (Output {df_metriche['RMSE %'].idxmin()})")

# Mostra alcuni esempi
# Crea il DataFrame con TUTTI e 15 gli output
print(f"\nPrime 10 righe (confronto per ogni output):\n")

n_esempi = 10
for idx in range(n_esempi):
    print(f"\n--- Riga dataset {idx} ---")
    for i in range(y_test_orig.shape[1]):
        y_true_val = y_test_orig[idx, i]
        y_pred_val = y_pred[idx, i]
        errore = abs(y_true_val - y_pred_val)
        print(f"  Output {i:2d}: y_true={y_true_val:10.4f} | y_pred={y_pred_val:10.4f} | Errore={errore:8.4f}")



# CELL 9 - Valutazioni del modello
print("\n=== VALUTAZIONE MODELLO ===")
print(f"✓ Epoca di stop: {len(history.history['loss'])} (buono se < 500)")
print(f"✓ R² Score: {r2:.4f} (buono se > 0.85)")
print(f"✓ RMSE: {rmse:.6f} ({rmse_percentage:.2f}% del range)")
print(f"✓ Val Loss: {val_loss_finale:.6f}")
print(f"✓ Test Loss: {test_loss:.6f}")
print(f"✓ Differenza: {abs(test_loss - val_loss_finale):.6f} (buono se < 0.01)")

if r2 > 0.85 and rmse_percentage < 10:
    print("\n✓✓✓ MODELLO ECCELLENTE ✓✓✓")
elif r2 > 0.7 and rmse_percentage < 15:
    print("\n✓✓ MODELLO BUONO ✓✓")
else:
    print("\n✗ MODELLO DA MIGLIORARE ✗")

# Cell 9 - salva modello e scaler
model.save(os.path.join('models', 'final_model.keras'))  # Aggiungi .keras
joblib.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, os.path.join('models', 'scalers.joblib'))
print("Modello e scaler salvati in `models/`")


# python
def plot_model_evaluation(history, val_loss_finale, test_loss, rmse, rmse_percentage, r2, df_metriche):
    """
    Genera grafici per la valutazione completa del modello.
    """

    # 4. Statistiche Globali - RMSE % per ogni output
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Statistiche Globali: RMSE % per Output', fontsize=14, fontweight='bold')

    outputs = df_metriche['Output'].values
    rmse_perc = df_metriche['RMSE %'].values
    colors_rmse = ['#2ecc71' if x < 10 else '#f39c12' if x < 15 else '#e74c3c' for x in rmse_perc]

    bars = ax.bar(outputs, rmse_perc, color=colors_rmse, edgecolor='black', linewidth=1.5, width=0.7)
    ax.set_xlabel('Output', fontsize=12)
    ax.set_ylabel('RMSE %', fontsize=12)
    ax.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Soglia Eccellente (10%)')
    ax.axhline(y=15, color='orange', linestyle='--', linewidth=2, label='Soglia Buono (15%)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10)

    for bar, val in zip(bars, rmse_perc):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # 5. R² per ogni output
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Statistiche Globali: R² Score per Output', fontsize=14, fontweight='bold')

    outputs = df_metriche['Output'].values
    r2_scores = df_metriche['R²'].values
    colors_r2 = ['#27ae60' if x > 0.85 else '#f39c12' if x > 0.7 else '#e74c3c' for x in r2_scores]

    bars = ax.bar(outputs, r2_scores, color=colors_r2, edgecolor='black', linewidth=1.5, width=0.7)
    ax.set_xlabel('Output', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.85, color='green', linestyle='--', linewidth=2, label='Soglia Eccellente (0.85)')
    ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, label='Soglia Buono (0.7)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=10)

    for bar, val in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # 6. Valutazione Modello - Metriche a confronto
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Valutazione Modello - Metriche Riassuntive', fontsize=14, fontweight='bold')

    metriche_nomi = ['R² medio', 'RMSE %\nmedio', 'Val Loss', 'Test Loss']
    metriche_valori = [
        df_metriche['R²'].mean(),
        df_metriche['RMSE %'].mean() / 100,  # Normalizza per visualizzazione
        val_loss_finale,
        test_loss
    ]

    # Normalizza per visualizzazione comparativa
    metriche_norm = [
        df_metriche['R²'].mean(),  # 0-1
        min(df_metriche['RMSE %'].mean() / 20, 1),  # Scala RMSE %
        min(val_loss_finale * 100, 1),  # Scala loss
        min(test_loss * 100, 1)  # Scala loss
    ]

    colors_metriche = ['#27ae60', '#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(metriche_nomi, metriche_norm, color=colors_metriche, edgecolor='black', linewidth=2, width=0.6)

    ax.set_ylabel('Valore Normalizzato', fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # Aggiungi etichette con valori reali
    labels_reali = [
        f'{metriche_valori[0]:.4f}',
        f'{df_metriche["RMSE %"].mean():.2f}%',
        f'{metriche_valori[2]:.6f}',
        f'{metriche_valori[3]:.6f}'
    ]

    for bar, label in zip(bars, labels_reali):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()


# Chiama la funzione alla fine dello script
plot_model_evaluation(history, val_loss_finale, test_loss, rmse, rmse_percentage, r2, df_metriche)


# python
def plot_predictions_vs_actual(y_test_orig, y_pred, output_dim):
    """
    Genera grafici con bisettrice per ogni output.
    Mostra come le predizioni si distribuiscono rispetto ai valori reali.
    """

    # Calcola il numero di righe e colonne per il subplot
    n_cols = 4
    n_rows = (output_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()  # Appiattisci per iterare facilmente

    fig.suptitle('Predizioni vs Valori Reali (con bisettrice)', fontsize=16, fontweight='bold')

    for i in range(output_dim):
        ax = axes[i]

        y_true = y_test_orig[:, i]
        y_pred_i = y_pred[:, i]

        # Calcola metriche
        mse = mean_squared_error(y_true, y_pred_i)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred_i)

        # Scatter plot
        ax.scatter(y_true, y_pred_i, alpha=0.5, s=30, color='#3498db', edgecolors='black', linewidth=0.5)

        # Bisettrice (linea perfetta)
        min_val = min(y_true.min(), y_pred_i.min())
        max_val = max(y_true.max(), y_pred_i.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predizione perfetta', zorder=5)

        # Formatting
        ax.set_xlabel('Valore Reale', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predizione', fontsize=11, fontweight='bold')
        ax.set_title(f'Output {i}: R²={r2:.4f}, RMSE={rmse:.4f}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)

        # Aggiungi statistiche nel grafico
        stats_text = f"n={len(y_true)}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Nascondi gli assi vuoti
    for j in range(output_dim, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# Chiama la funzione
plot_predictions_vs_actual(y_test_orig, y_pred, y_test_orig.shape[1])

