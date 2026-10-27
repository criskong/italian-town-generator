"""
Passo 2 - Rete neurale scritta da zero con NumPy.

La rete più semplice possibile: un solo strato di pesi.
Riceve una lettera e predice le probabilità della lettera successiva,
come il modello a bigrammi. La differenza è che i numeri non li contiamo noi:
la rete li impara da sola, abbassando la loss con la discesa del gradiente.
"""

import numpy as np

from bigrammi import FILE_DATI, SEGNALE, carica_nomi, costruisci_vocabolario

SEME = 42
PASSI_DI_ADDESTRAMENTO = 300
LEARNING_RATE = 50.0
STAMPA_OGNI = 25
NUMERO_NOMI = 20
LUNGHEZZA_MASSIMA = 50


# ---------------------------------------------------------------------------
# Preparazione dei dati
# ---------------------------------------------------------------------------

def crea_dataset(nomi: list[str], stoi: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Trasforma i nomi in coppie (lettera attuale -> lettera successiva), come numeri.
    "napoli" diventa: (.->n), (n->a), (a->p), (p->o), (o->l), (l->i), (i->.)
    """
    ingressi = []
    uscite = []
    for nome in nomi:
        lettere = [SEGNALE] + list(nome) + [SEGNALE]
        for a, b in zip(lettere, lettere[1:]):
            ingressi.append(stoi[a])
            uscite.append(stoi[b])
    return np.array(ingressi), np.array(uscite)


def one_hot(indici: np.ndarray, dimensione: int) -> np.ndarray:
    """
    Trasforma ogni numero in un vettore di zeri con un solo 1.
    Con 5 simboli, il numero 2 diventa [0, 0, 1, 0, 0].
    """
    matrice = np.zeros((len(indici), dimensione))
    matrice[np.arange(len(indici)), indici] = 1.0
    return matrice


# ---------------------------------------------------------------------------
# I mattoni della rete
# ---------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Trasforma numeri qualsiasi (anche negativi) in probabilità che sommano a 1.
    Sottrarre il massimo non cambia il risultato ma evita overflow con exp().
    """
    logits = logits - logits.max(axis=1, keepdims=True)
    esponenziali = np.exp(logits)
    return esponenziali / esponenziali.sum(axis=1, keepdims=True)


def calcola_loss(probabilita: np.ndarray, uscite: np.ndarray) -> float:
    """Stessa loss del Passo 1: sorpresa media davanti alle lettere vere."""
    probabilita_corrette = probabilita[np.arange(len(uscite)), uscite]
    return float(-np.log(probabilita_corrette).mean())


# ---------------------------------------------------------------------------
# Addestramento
# ---------------------------------------------------------------------------

def addestra(X: np.ndarray, Y: np.ndarray, dimensione: int, rng: np.random.Generator) -> np.ndarray:
    # I pesi: una tabella 39 x 39 di numeri casuali. È tutta la "conoscenza" della rete.
    W = rng.standard_normal((dimensione, dimensione))
    n = len(Y)

    for passo in range(PASSI_DI_ADDESTRAMENTO):
        # 1. FORWARD: dalle lettere alle probabilità
        logits = X @ W
        probabilita = softmax(logits)
        loss = calcola_loss(probabilita, Y)

        # 2. BACKWARD: in che direzione spostare ogni peso per abbassare la loss
        dlogits = probabilita.copy()
        dlogits[np.arange(n), Y] -= 1
        dlogits /= n
        dW = X.T @ dlogits

        # 3. UPDATE: un piccolo passo "in discesa"
        W -= LEARNING_RATE * dW

        if passo % STAMPA_OGNI == 0:
            print(f"  passo {passo:4d}  |  loss {loss:.4f}")

    loss_finale = calcola_loss(softmax(X @ W), Y)
    print(f"  finale     |  loss {loss_finale:.4f}")
    return W


# ---------------------------------------------------------------------------
# Uso della rete addestrata
# ---------------------------------------------------------------------------

def probabilita_dopo(lettera: str, W: np.ndarray, stoi: dict[str, int]) -> np.ndarray:
    x = one_hot(np.array([stoi[lettera]]), len(stoi))
    return softmax(x @ W)[0]


def stampa_previsioni(lettera: str, W: np.ndarray, stoi: dict[str, int], itos: list[str]) -> None:
    probabilita = probabilita_dopo(lettera, W, stoi)
    migliori = np.argsort(probabilita)[::-1][:3]
    previsioni = ", ".join(f"'{itos[i]}' {probabilita[i]:.1%}" for i in migliori)
    print(f"  dopo '{lettera}': {previsioni}")


def genera_nome(W: np.ndarray, stoi: dict[str, int], itos: list[str], rng: np.random.Generator) -> str:
    lettere = []
    lettera = SEGNALE
    while len(lettere) < LUNGHEZZA_MASSIMA:
        probabilita = probabilita_dopo(lettera, W, stoi)
        indice = rng.choice(len(itos), p=probabilita)
        if indice == 0:
            break
        lettera = itos[indice]
        lettere.append(lettera)
    return "".join(lettere)


def main() -> None:
    rng = np.random.default_rng(SEME)

    nomi = carica_nomi(FILE_DATI)
    stoi, itos = costruisci_vocabolario(nomi)
    ingressi, uscite = crea_dataset(nomi, stoi)
    X = one_hot(ingressi, len(stoi))

    print(f"Esempi di addestramento: {len(uscite)} coppie di lettere\n")
    print("Addestramento:")
    W = addestra(X, uscite, len(stoi), rng)

    print("\nCosa ha imparato la rete:")
    for lettera in ["q", ".", "z"]:
        stampa_previsioni(lettera, W, stoi, itos)

    nomi_reali = set(nomi)
    print(f"\n{NUMERO_NOMI} nomi generati:")
    generati = 0
    while generati < NUMERO_NOMI:
        nome = genera_nome(W, stoi, itos, rng)
        if not nome:
            continue
        nota = "  (esiste davvero!)" if nome in nomi_reali else ""
        print(f"  {nome.title()}{nota}")
        generati += 1


if __name__ == "__main__":
    main()