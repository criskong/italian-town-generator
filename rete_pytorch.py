"""
Passo 3 - Rete neurale con PyTorch.

Rispetto al Passo 2 la rete:
- guarda le ultime 3 lettere invece di una sola (più contesto);
- rappresenta ogni lettera con un embedding imparato invece che con il one-hot;
- ha uno strato nascosto di neuroni con attivazione tanh;
- viene valutata su nomi che non ha mai visto (validation e test).
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from bigrammi import FILE_DATI, SEGNALE, carica_nomi, costruisci_vocabolario

SEME = 42

# Architettura della rete
CONTESTO = 3            # quante lettere precedenti guarda la rete
DIM_EMBEDDING = 10      # quanti numeri rappresentano ogni lettera
NEURONI_NASCOSTI = 200  # dimensione dello strato nascosto

# Addestramento
DIM_BATCH = 32
PASSI = 30_000
LR_INIZIALE = 0.1
LR_FINALE = 0.01
PASSO_CAMBIO_LR = 20_000
STAMPA_OGNI = 2_000

# Generazione
NUMERO_NOMI = 20
LUNGHEZZA_MASSIMA = 50


# ---------------------------------------------------------------------------
# Preparazione dei dati
# ---------------------------------------------------------------------------

def dividi_nomi(nomi: list[str], rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    """Mescola i nomi e li divide in training (80%), validation (10%) e test (10%)."""
    nomi = nomi.copy()
    rng.shuffle(nomi)
    fine_training = int(0.8 * len(nomi))
    fine_validation = int(0.9 * len(nomi))
    return nomi[:fine_training], nomi[fine_training:fine_validation], nomi[fine_validation:]


def crea_dataset(nomi: list[str], stoi: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per ogni lettera di ogni nome crea un esempio:
    ingresso = le 3 lettere precedenti, risposta = la lettera attuale.
    All'inizio il contesto è riempito con il segnale (indice 0).
    """
    ingressi = []
    uscite = []
    for nome in nomi:
        contesto = [0] * CONTESTO
        for carattere in list(nome) + [SEGNALE]:
            indice = stoi[carattere]
            ingressi.append(contesto)
            uscite.append(indice)
            contesto = contesto[1:] + [indice]  # fa scorrere la finestra di una lettera
    return torch.tensor(ingressi), torch.tensor(uscite)


# ---------------------------------------------------------------------------
# La rete
# ---------------------------------------------------------------------------

class GeneratoreComuni(nn.Module):
    def __init__(self, dim_vocabolario: int):
        super().__init__()
        self.embedding = nn.Embedding(dim_vocabolario, DIM_EMBEDDING)
        self.nascosto = nn.Linear(CONTESTO * DIM_EMBEDDING, NEURONI_NASCOSTI)
        self.uscita = nn.Linear(NEURONI_NASCOSTI, dim_vocabolario)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                 # (batch, 3, 10): 3 lettere, 10 numeri ciascuna
        emb = emb.view(emb.shape[0], -1)        # (batch, 30): le 3 lettere messe in fila
        h = torch.tanh(self.nascosto(emb))      # (batch, 200): lo strato nascosto
        return self.uscita(h)                   # (batch, 39): un logit per ogni simbolo


# ---------------------------------------------------------------------------
# Addestramento e valutazione
# ---------------------------------------------------------------------------

@torch.no_grad()
def valuta(modello: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    """Loss su un intero gruppo di dati, senza addestrare."""
    return F.cross_entropy(modello(X), Y).item()


def addestra(modello: nn.Module, X_tr, Y_tr, X_val, Y_val, generatore: torch.Generator) -> None:
    ottimizzatore = torch.optim.SGD(modello.parameters(), lr=LR_INIZIALE)

    for passo in range(PASSI):
        # Minibatch: 32 esempi a caso invece di tutti
        indici = torch.randint(0, len(X_tr), (DIM_BATCH,), generator=generatore)

        # 1. FORWARD
        logits = modello(X_tr[indici])
        loss = F.cross_entropy(logits, Y_tr[indici])

        # 2. BACKWARD: PyTorch calcola da solo tutti i gradienti
        ottimizzatore.zero_grad()
        loss.backward()

        # 3. UPDATE: per ogni peso, peso -= learning_rate * gradiente
        ottimizzatore.step()

        if passo == PASSO_CAMBIO_LR:
            for gruppo in ottimizzatore.param_groups:
                gruppo["lr"] = LR_FINALE

        if passo % STAMPA_OGNI == 0:
            loss_tr = valuta(modello, X_tr, Y_tr)
            loss_val = valuta(modello, X_val, Y_val)
            print(f"  passo {passo:6d}  |  training {loss_tr:.4f}  |  validation {loss_val:.4f}")


# ---------------------------------------------------------------------------
# Generazione
# ---------------------------------------------------------------------------

@torch.no_grad()
def genera_nome(modello: nn.Module, itos: list[str], generatore: torch.Generator) -> str:
    contesto = [0] * CONTESTO
    lettere = []
    while len(lettere) < LUNGHEZZA_MASSIMA:
        logits = modello(torch.tensor([contesto]))
        probabilita = F.softmax(logits, dim=1)
        indice = torch.multinomial(probabilita, num_samples=1, generator=generatore).item()
        if indice == 0:
            break
        lettere.append(itos[indice])
        contesto = contesto[1:] + [indice]
    return "".join(lettere)


def main() -> None:
    torch.manual_seed(SEME)                         # pesi iniziali riproducibili
    generatore = torch.Generator().manual_seed(SEME)
    rng = random.Random(SEME)

    nomi = carica_nomi(FILE_DATI)
    stoi, itos = costruisci_vocabolario(nomi)

    nomi_tr, nomi_val, nomi_test = dividi_nomi(nomi, rng)
    X_tr, Y_tr = crea_dataset(nomi_tr, stoi)
    X_val, Y_val = crea_dataset(nomi_val, stoi)
    X_test, Y_test = crea_dataset(nomi_test, stoi)

    print(f"Nomi: training {len(nomi_tr)}  |  validation {len(nomi_val)}  |  test {len(nomi_test)}")
    print(f"Esempi di training: {len(Y_tr)}")

    modello = GeneratoreComuni(len(itos))
    numero_parametri = sum(p.numel() for p in modello.parameters())
    print(f"Parametri della rete: {numero_parametri}\n")

    print("Addestramento:")
    addestra(modello, X_tr, Y_tr, X_val, Y_val, generatore)

    print("\nRisultati finali:")
    print(f"  training   {valuta(modello, X_tr, Y_tr):.4f}")
    print(f"  validation {valuta(modello, X_val, Y_val):.4f}")
    print(f"  test       {valuta(modello, X_test, Y_test):.4f}")

    nomi_reali = set(nomi)
    print(f"\n{NUMERO_NOMI} nomi generati:")
    copiati = 0
    generati = 0
    while generati < NUMERO_NOMI:
        nome = genera_nome(modello, itos, generatore)
        if not nome:
            continue
        if nome in nomi_reali:
            copiati += 1
            nota = "  (esiste davvero!)"
        else:
            nota = ""
        print(f"  {nome.title()}{nota}")
        generati += 1

    print(f"\nNomi già esistenti: {copiati} su {NUMERO_NOMI}")


if __name__ == "__main__":
    main()