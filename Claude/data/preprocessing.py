import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('database.dat')

df = pd.DataFrame(data)

print(f"Shape del dataset: {df.shape}")

print(f"\nInformazioni dataset:")
print(df.info())


# Verifica valori mancanti
"""print("Analisi Valori Mancanti:")
print(df.isnull().sum())
print(f"\nRighe duplicate: {df.duplicated().sum()}")"""

# Se ci sono valori mancanti, rimuoviamoli
df_cleaned_compl = df.dropna()   # è una funziona che salta le righe dove ci sono valori mancanti (dropna = drop + nane)
print(f"\n✓ Dataset dopo pulizia: {df_cleaned_compl.shape}")
print(f"Righe rimosse: {len(df) - len(df_cleaned_compl)}")

# Separazione input (X) e output (y) prima della rimozione degli outlier
dof_columns = [col for col in df_cleaned_compl.columns if col.startswith('DOF')]
of_columns = [col for col in df_cleaned_compl.columns if col.startswith('OF')]

# Matrice di correlazione prima della rimozione degli outlier
cols_to_remove = list(df_cleaned_compl.columns[:2])

df_reduced = df_cleaned_compl.drop(columns=cols_to_remove, errors='ignore')

correlation_reduced_compl = df_reduced.corr()
correlation_cross = correlation_reduced_compl.loc[dof_columns, of_columns]

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_cross, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Matrice di Correlazione tra DOF e OF (senza rimozione outlier)')
plt.tight_layout()
plt.show()



X_compl = df_cleaned_compl[dof_columns].values  # è una lista formata da 922 liste che all'interno contengono 7 valori (input = feature) (DOF1, DOF2, DOF3, DOF4, DOF5, DOF6, DOF7)
y_compl = df_cleaned_compl[of_columns].values   # è una lista formata da 922 liste che all'interno contengono 15 valori (output = target) (OF1, OF2, OF3, OF4, OF5, OF6, OF7, OF8, OF9, OF10, OF11, OF12, OF13, OF14, OF15)

#Divisione del dataset in training e test set (80% train, 20% test)
# Primo split: 70% train, 30% temp
X_train_compl, X_temp_compl, y_train_compl, y_temp_compl = train_test_split(X_compl, y_compl, test_size=0.3, random_state=42)

# Secondo split: Dividi il 30% in 15% val e 15% test
X_val_compl, X_test_compl, y_val_compl, y_test_compl = train_test_split(X_temp_compl, y_temp_compl, test_size=0.5, random_state=42)

# Creiamo gli scaler, standardizzazione Z-score
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# REGOLA D'ORO: fit_transform SOLO sul dataset di addestramento
X_scaled_compl = scaler_X.fit_transform(X_train_compl)
y_scaled_compl = scaler_y.fit_transform(y_train_compl)

X_val_scaled_compl   = scaler_X.transform(X_val_compl)
X_test_scaled_compl  = scaler_X.transform(X_test_compl)

y_val_scaled_compl   = scaler_y.transform(y_val_compl)
y_test_scaled_compl  = scaler_y.transform(y_test_compl)

# Rimozione Outlier con metodo IQR
def remove_outliers_iqr(data, columns, multiplier=1.5):

    df_out = data.copy()

    # Salviamo la lunghezza iniziale per calcolare correttamente quante righe sono state rimosse
    initial_len = len(df_out)

    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Filtriamo mantenendo solo le righe dentro i bound per la colonna corrente
        df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]

    # Calcolo righe rimosse come differenza tra prima e dopo (evita doppio conteggio)
    rows_removed = initial_len - len(df_out)

    return df_out, rows_removed

numeric_columns = df_cleaned_compl.select_dtypes(include=[np.number]).columns.tolist() # Seleziona solo le colonne numeriche per il controllo degli outlier

# Applica rimozione outlier
df_cleaned, outliers_removed = remove_outliers_iqr(df_cleaned_compl, numeric_columns)

#Dopo questa funzione di rimozione degli outlier il database non è più df ma df_cleaned.
#La funzione infatti restituisce il dataset (df_out) e queste (nel passaggio sopra) viene salvato in df_cleaned.
#Quando una funzione restituisce 2 valori o più, è necessario salvarli in variabili separate
#(in questo caso df_cleaned e outliers_removed) per poterli utilizzare successivamente.

pd.set_option('display.max_columns', None)

print(f"\n✓ Outlier rilevati e rimossi: {outliers_removed}")
print(f"\n✓ Dataset dopo rimozione outlier: {df_cleaned.shape}")
print(f"\nInformazioni dataset:")
print(df_cleaned.info())

# Identifica colonne DOF (input) e OF (output)
dof_columns = [col for col in df_cleaned.columns if col.startswith('DOF')]
of_columns = [col for col in df_cleaned.columns if col.startswith('OF')]

print(f"\n Colonne DOF (INPUT - {len(dof_columns)}):")
print(dof_columns)

print(f"\n Colonne OF (OUTPUT - {len(of_columns)}):")
print(of_columns)

# Matrice di correlazione prima della rimozione degli outlier
cols_to_remove = list(df_cleaned.columns[:2])

df_reduced = df_cleaned.drop(columns=cols_to_remove, errors='ignore')

correlation_reduced = df_reduced.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_reduced, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Matrice di Correlazione tra DOF e OF (Con rimozione outlier)')
plt.tight_layout()
plt.show()

print("\n Correlazioni più forti tra DOF e OF:")
for of_col in of_columns:
    correlations = correlation_reduced.loc[dof_columns, of_col].sort_values(key=abs, ascending=False)
    print(f"\n{of_col}:")
    for dof, corr in correlations.items():
        print(f"  {dof}: {corr:.4f}")


# Separazione input (X) e output (y)
X = df_cleaned[dof_columns].values
y = df_cleaned[of_columns].values

# ================================================================================================================
print("\n" + "="*100)
print("Analisi prima di effettuare la rimozione outlier")
print("="*100)

print(f"\nShape dei dati(senza rimozione outlier):")
print(f"  X (Input): {X_compl.shape}")
print(f"  y (Output): {y_compl.shape}")

def print_stats_per_column(col_names, original_array, scaled_array, name_prefix=""):
    print(f"\nStatistiche per {name_prefix} (per colonna):")
    nome_colmn = []
    min_orig = []
    min_sca = []
    max_orig = []
    max_sca = []
    mean_orig = []
    mean_sca = []
    std_orig = []
    std_sca = []
    for i, col in enumerate(col_names):
        nome_colmn.append(col)
        orig = original_array[:, i]
        min_orig.append(orig.min())
        max_orig.append(orig.max())
        mean_orig.append(orig.mean())
        std_orig.append(orig.std())
        sca = scaled_array[:, i]
        min_sca.append(sca.min())
        max_sca.append(sca.max())
        mean_sca.append(sca.mean())
        std_sca.append(sca.std())

    np.set_printoptions(precision=3, suppress=True, linewidth=200)

    # Per pandas: mostra tutte le colonne, evita il wrapping e imposta formato float
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)  # aumenta la larghezza di stampa
    pd.set_option('display.expand_frame_repr', False)  # evita che il DataFrame vada a capo
    pd.options.display.float_format = '{:.3f}'.format
    tab = pd.DataFrame({
         'Nome': nome_colmn,
        'Min orig': min_orig, 'Max orig': max_orig, 'Mean orig': mean_orig, 'Std orig': std_orig, '|': '|',
        'Min sca': min_sca, 'Max sca': max_sca, 'Mean sca': mean_sca, 'Std sca': std_sca
    })

    print(tab)


# Esempio di utilizzo (metti alla fine della sezione di normalizzazione, dove hai già `dof_columns`, `of_columns`, `X`, `y`, `X_scaled`, `y_scaled`)
print_stats_per_column(dof_columns, X_compl, X_scaled_compl, name_prefix="DOF (senza rimozione outlier)")
print_stats_per_column(of_columns,  y_compl, y_scaled_compl,  name_prefix="OF (senza rimozione outlier)")

print(f"\nSplit completato (senza rimozione outlier):")
print(f"\n  🔵 Training Set: {X_train_compl.shape[0]} campioni ({X_train_compl.shape[0]/len(X_scaled_compl)*100:.1f}%)")
print(f"  🟡 Validation Set: {X_val_compl.shape[0]} campioni ({X_val_compl.shape[0]/len(X_scaled_compl)*100:.1f}%)")
print(f"  🔴 Test Set: {X_test_compl.shape[0]} campioni ({X_test_compl.shape[0]/len(X_scaled_compl)*100:.1f}%)")
print(f"\n  Totale: {X_train_compl.shape[0] + X_val_compl.shape[0] + X_test_compl.shape[0]} campioni")

print("\n" + "="*100)
print("Analisi dopo aver effettuato la rimozione outlier")
print("="*100)

print(f"\nShape dei dati:")
print(f"  X (Input): {X.shape}")
print(f"  y (Output): {y.shape}")

#Divisione del dataset in training e test set (80% train, 20% test)
# Primo split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Secondo split: Dividi il 30% in 15% val e 15% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Creiamo gli scaler, standardizzazione Z-score
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# REGOLA D'ORO: fit_transform SOLO sul dataset di addestramento
X_scaled = scaler_X.fit_transform(X_train)
y_scaled = scaler_y.fit_transform(y_train)

X_val_scaled   = scaler_X.transform(X_val)
X_test_scaled  = scaler_X.transform(X_test)

y_val_scaled   = scaler_y.transform(y_val)
y_test_scaled  = scaler_y.transform(y_test)

print_stats_per_column(dof_columns, X, X_scaled, name_prefix="DOF (con rimozione outlier)")
print_stats_per_column(of_columns,  y, y_scaled,  name_prefix="OF (con rimozione outlier)")

print(f"\nSplit completato:")
print(f"\n  🔵 Training Set: {X_train.shape[0]} campioni ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"  🟡 Validation Set: {X_val.shape[0]} campioni ({X_val.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"  🔴 Test Set: {X_test.shape[0]} campioni ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"\n  Totale: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} campioni")




'==================================================================================================================================================='
# Creazione e stampaggio dei grafici (va sistemata, ottimizata, ma funziona!!!!!!!!!!!!!)
'==================================================================================================================================================='
"""# Calcola statistiche per ogni colonna prima della normalizzazione (senza rimozione outlier)
def calcola_statistiche(data, colonne):
    stats = []
    for i in range(data.shape[1]):
        stats.append({
            'nome': colonne[i],
            'min': data[:, i].min(),
            'max': data[:, i].max(),
            'mean': data[:, i].mean(),
            'std': data[:, i].std()
        })
    return stats


stats_X = calcola_statistiche(X_compl, dof_columns)
stats_y = calcola_statistiche(y_compl, of_columns)


def disegna_subplot(ax, stats, titolo, colore_sfondo):
    nomi = [s['nome'] for s in stats]
    mins = [s['min'] for s in stats]
    maxs = [s['max'] for s in stats]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]

    x = np.arange(len(nomi))
    width = 0.2

    ax.bar(x - 1.5 * width, mins, width, label='Min', color='#e74c3c', alpha=0.85)
    ax.bar(x - 0.5 * width, maxs, width, label='Max', color='#2ecc71', alpha=0.85)
    ax.bar(x + 0.5 * width, means, width, label='Media', color='#3498db', alpha=0.85)
    ax.bar(x + 1.5 * width, stds, width, label='Std', color='#f39c12', alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_facecolor(colore_sfondo)
    ax.set_xticks(x)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticklabels(nomi, rotation=45, ha='right', fontsize=13)
    ax.set_ylabel('Valore (scala standardizzata)', fontsize=10)
    ax.set_title(titolo, fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)


# ── Figura con 2 subplot verticali ───────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1,
                               figsize=(max(12, max(len(dof_columns), len(of_columns)) * 1.2), 12), gridspec_kw={'hspace': 0.7})
fig.subplots_adjust(top=0.88, bottom=0.22)

disegna_subplot(ax1, stats_X,
                'Statistiche pre-normalizzazione — DOF (Input)','#f0f4ff')

disegna_subplot(ax2, stats_y,
                'Statistiche pre-normalizzazione — OF (Output)','#fff8f0')
plt.show()


stats_X = calcola_statistiche(X_scaled_compl, dof_columns)
stats_y = calcola_statistiche(y_scaled_compl, of_columns)


# ── Figura con 2 subplot verticali ───────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1,
                               figsize=(max(12, max(len(dof_columns), len(of_columns)) * 1.2), 12), gridspec_kw={'hspace': 0.7})
fig.subplots_adjust(top=0.88, bottom=0.22)

disegna_subplot(ax1, stats_X,
                'Statistiche post-normalizzazione — DOF (Input)','#f0f4ff')

disegna_subplot(ax2, stats_y,
                'Statistiche post-normalizzazione — OF (Output)','#fff8f0')
plt.show()


stats_X = calcola_statistiche(X, dof_columns)
stats_y = calcola_statistiche(y, of_columns)

# ── Figura con 2 subplot verticali ───────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1,
                               figsize=(max(12, max(len(dof_columns), len(of_columns)) * 1.2), 12), gridspec_kw={'hspace': 0.7})
fig.subplots_adjust(top=0.88, bottom=0.22)

disegna_subplot(ax1, stats_X,
                'Statistiche pre-normalizzazione — DOF (Input) (con rimozione outlier)','#f0f4ff')

disegna_subplot(ax2, stats_y,
                'Statistiche pre-normalizzazione — OF (Output)','#fff8f0')
plt.show()


stats_X = calcola_statistiche(X_scaled_compl, dof_columns)
stats_y = calcola_statistiche(y_scaled_compl, of_columns)


# ── Figura con 2 subplot verticali ───────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1,
                               figsize=(max(12, max(len(dof_columns), len(of_columns)) * 1.2), 12), gridspec_kw={'hspace': 0.7})
fig.subplots_adjust(top=0.88, bottom=0.22)

disegna_subplot(ax1, stats_X,
                'Statistiche post-normalizzazione — DOF (Input) (con rimozione outlier)','#f0f4ff')

disegna_subplot(ax2, stats_y,
                'Statistiche post-normalizzazione — OF (Output)','#fff8f0')
plt.show()

def plot_hist_grid(data, col_names, title, cols=5, bins=30, figsize=(15, 9), dpi=150):
    
    n_plots = len(col_names)
    rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # rende 1D anche se rows*cols == 1

    for i, name in enumerate(col_names):
        ax = axes[i]
        ax.hist(data[:, i], bins=bins, color='#4c72b0', edgecolor='black', alpha=0.85)
        ax.set_title(name, fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(axis='y', alpha=0.25)

    # rimuove eventuali assi vuoti
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    plt.show()
    plt.close(fig)


# Istogrammi DOF e OF prima della normalizzazione (7 DOF, 15 OF)
plot_hist_grid(X_compl, dof_columns, 'Distribuzioni DOF - prima normalizzazione (dataset intero)', cols=4)
plot_hist_grid(y_compl, of_columns, 'Distribuzioni OF - prima normalizzazione (dataset intero)', cols=5)

# Istogrammi DOF e OF dopo la normalizzazione (7 DOF, 15 OF)
plot_hist_grid(X_scaled_compl, dof_columns, 'Distribuzioni DOF - dopo normalizzazione (dataset intero)', cols=4)
plot_hist_grid(y_scaled_compl, of_columns, 'Distribuzioni OF - dopo normalizzazione (dataset intero)', cols=5)

# Istogrammi DOF e OF prima della normalizzazione (7 DOF, 15 OF)
plot_hist_grid(X, dof_columns, 'Distribuzioni DOF - prima normalizzazione (con rimozione outlier)', cols=4)
plot_hist_grid(y, of_columns, 'Distribuzioni OF - prima normalizzazione (con rimozione outlier)', cols=5)

# Istogrammi DOF e OF dopo la normalizzazione (7 DOF, 15 OF)
plot_hist_grid(X_scaled, dof_columns, 'Distribuzioni DOF - dopo normalizzazione (con rimozione outlier)', cols=4)
plot_hist_grid(y_scaled, of_columns, 'Distribuzioni OF - dopo normalizzazione (con rimozione outlier)', cols=5)"""