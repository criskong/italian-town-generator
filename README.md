# Italian Town Generator

Un'intelligenza artificiale generativa che inventa nomi di comuni italiani
che non esistono, imparando lettera per lettera dai circa 7.900 comuni reali.

Progetto didattico costruito in tappe, dal modello più semplice alla rete neurale:

- [x] **Passo 0** – Preparazione dei dati → [spiegazione](docs/passo-0-dati.md)
- [x] **Passo 1** – Modello a bigrammi (Python puro) → [spiegazione](docs/passo-1-bigrammi.md)
- [x] **Passo 2** – Rete neurale scritta da zero con NumPy → [spiegazione](docs/passo-2-rete-numpy.md)
- [ ] **Passo 3** – Rete neurale con PyTorch

## Risultati

| Modello | Loss | Perplessità | Esempi generati |
|---|---|---|---|
| Casuale (riferimento) | 3,664 | 39,0 | – |
| Bigrammi | 2,260 | 9,6 | Cilano, Miascano, Velno |
| Rete neurale NumPy (1 strato) | 2,268 | 9,7 | Gintenda Bulona, Ma Gioro, Grardontemo |

La **loss** misura quanto il modello è "sorpreso" dai nomi reali: più bassa è, meglio è.
La **perplessità** (`e^loss`) indica tra quante lettere, in media, il modello è indeciso.

La rete del Passo 2, partendo da pesi casuali, arriva da sola agli stessi risultati
dei conteggi del Passo 1: vede una sola lettera, quindi non può fare di meglio.
Il Passo 3 le darà più contesto.

## Come provarlo

```bash
git clone https://github.com/criskong/italian-town-generator.git
cd italian-town-generator
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
python bigrammi.py
python rete_numpy.py
```

Per rigenerare i dati, scarica l'[elenco dei comuni ISTAT](https://www.istat.it/storage/codici-unita-amministrative/Elenco-comuni-italiani.xlsx),
salvalo in `data/Elenco-comuni-italiani.xlsx` e lancia:

```bash
python prepara_dati.py
```

## Dati

Fonte: ISTAT – Codici delle unità amministrative territoriali.