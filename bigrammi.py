"""
Passo 1 - Modello a bigrammi.

Il modello più semplice di AI generativa: conta per ogni lettera,
quale lettera la segue nei nomi reali, poi usa quei conteggi
come probabilità per generare nomi nuovi.
"""

import math
import random
from pathlib import Path

FILE_DATI = Path("data") / "comuni.txt"
SEGNALE = "."      # segna sia l'inizio che la fine di un nome
SEME = 42          # rende i risultati riproducibili
NUMERO_NOMI = 20
LUNGHEZZA_MASSIMA = 50


def carica_nomi(percorso: Path) -> list[str]:
    return percorso.read_text(encoding="utf-8").splitlines()


def costruisci_vocabolario(nomi: list[str]) -> tuple[dict[str, int], list[str]]:
    """
    Assegna un numero a ogni carattere.
    itos = "index to string" (numero -> carattere)
    stoi = "string to index" (carattere -> numero)
    """
    caratteri = sorted(set("".join(nomi)))
    if SEGNALE in caratteri:
        raise ValueError(f"Il carattere '{SEGNALE}' compare nei dati: scegli un altro segnale")

    itos = [SEGNALE] + caratteri   # il segnale ha sempre indice 0
    stoi = {carattere: indice for indice, carattere in enumerate(itos)}
    return stoi, itos


def conta_bigrammi(nomi: list[str], stoi: dict[str, int]) -> list[list[int]]:
    """
    Crea una tabella in cui conteggi[a][b] = quante volte la lettera b
    segue la lettera a in tutti i nomi.
    """
    n = len(stoi)
    conteggi = [[0] * n for _ in range(n)]

    for nome in nomi:
        lettere = [SEGNALE] + list(nome) + [SEGNALE]
        for a, b in zip(lettere, lettere[1:]):
            conteggi[stoi[a]][stoi[b]] += 1

    return conteggi


def calcola_probabilita(conteggi: list[list[int]], smoothing: int = 1) -> list[list[float]]:
    """
    Trasforma ogni riga di conteggi in probabilità (che sommano a 1).
    Lo smoothing aggiunge 1 a ogni conteggio, così nessuna coppia
    ha probabilità esattamente zero.
    """
    probabilita = []
    for riga in conteggi:
        riga_smussata = [c + smoothing for c in riga]
        totale = sum(riga_smussata)
        probabilita.append([c / totale for c in riga_smussata])
    return probabilita


def genera_nome(probabilita: list[list[float]], itos: list[str], rng: random.Random) -> str:
    """Parte dal segnale e pesca una lettera alla volta finché non esce di nuovo il segnale."""
    indice = 0
    lettere = []
    indici_possibili = range(len(itos))

    while len(lettere) < LUNGHEZZA_MASSIMA:
        indice = rng.choices(indici_possibili, weights=probabilita[indice])[0]
        if indice == 0:
            break
        lettere.append(itos[indice])

    return "".join(lettere)


def calcola_loss(nomi: list[str], probabilita: list[list[float]], stoi: dict[str, int]) -> float:
    """
    Negative log-likelihood media: misura quanto il modello è "sorpreso"
    dai nomi reali. Più è bassa, meglio il modello conosce i dati.
    """
    somma_log = 0.0
    numero_coppie = 0

    for nome in nomi:
        lettere = [SEGNALE] + list(nome) + [SEGNALE]
        for a, b in zip(lettere, lettere[1:]):
            p = probabilita[stoi[a]][stoi[b]]
            somma_log += math.log(p)
            numero_coppie += 1

    return -somma_log / numero_coppie


def stampa_lettere_iniziali(conteggi: list[list[int]], itos: list[str], quante: int = 5) -> None:
    riga = conteggi[0]  # la riga del segnale = come iniziano i nomi
    totale = sum(riga)
    migliori = sorted(range(len(riga)), key=lambda i: riga[i], reverse=True)[:quante]

    print("Lettere iniziali più frequenti:")
    for i in migliori:
        print(f"  '{itos[i]}': {riga[i] / totale:.1%}")


def main() -> None:
    nomi = carica_nomi(FILE_DATI)
    stoi, itos = costruisci_vocabolario(nomi)
    conteggi = conta_bigrammi(nomi, stoi)
    probabilita = calcola_probabilita(conteggi)

    print(f"Nomi: {len(nomi)}  |  Vocabolario: {len(itos)} simboli\n")
    stampa_lettere_iniziali(conteggi, itos)

    loss = calcola_loss(nomi, probabilita, stoi)
    loss_casuale = math.log(len(itos))
    print(f"\nLoss del modello:          {loss:.3f}")
    print(f"Loss di un modello a caso: {loss_casuale:.3f}")

    rng = random.Random(SEME)
    nomi_reali = set(nomi)
    print(f"\n{NUMERO_NOMI} nomi generati:")

    generati = 0
    while generati < NUMERO_NOMI:
        nome = genera_nome(probabilita, itos, rng)
        if not nome:
            continue
        nota = "  (esiste davvero!)" if nome in nomi_reali else ""
        print(f"  {nome.title()}{nota}")
        generati += 1


if __name__ == "__main__":
    main()