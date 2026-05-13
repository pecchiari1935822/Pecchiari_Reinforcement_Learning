# Parametri che descrivono lunghezza episodio e lunghezza del training
TOTAL_TIMESTEPS = 100_000
EPISODE_LENGTH = 20 # modifiche che possono essere fatte in un episodio
learning_rate = [0.00003]
n_steps = [200]

# Azione che viene fatta dall'attore che interagisce con l'ambiente
ACTION_SCALE   = 0.05

# Parametri che decrivono come è costruito il PPO
PPO_PARAMS = dict(
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
)

# Cosa si vuole ottimizzare (quali DOF) e in quale riga del file di input
ROW_INDEX = [3]  # INSERISCI L'INDICE DELLA RIGA CHE VUOI OTTIMIZZARE

ACTIVE_DOF_INDICES = [2,3,4,5,6]  # INSERISCI GLI INDICI DEI DOF CHE VUOI OTTIMIZZARE (0-5)

COPPIE_CUSTOM = [[1, 3], [2, 4], [5, 6]]

combinazioni_da_testare = []

# A. Aggiungi ogni DOF singolarmente
'''for idx in ACTIVE_DOF_INDICES:
    combinazioni_da_testare.append([idx])'''

# B. Aggiungi le coppie custom definite sopra
'''for coppia in COPPIE_CUSTOM:
    combinazioni_da_testare.append(coppia)'''

# C. Aggiungi tutti i DOF insieme (solo se sono più di 1, per evitare doppioni)
if len(ACTIVE_DOF_INDICES) > 1:
    combinazioni_da_testare.append(ACTIVE_DOF_INDICES)