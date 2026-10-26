# Italian Town Generator

Un'intelligenza artificiale generativa che inventa nomi di comuni italiani
che non esistono, imparando lettera per lettera dai circa 7.900 comuni reali.

Progetto didattico costruito in tappe, dal modello più semplice alla rete neurale:

- [x] **Passo 0** – Preparazione dei dati
- [ ] **Passo 1** – Modello a bigrammi (Python puro)
- [ ] **Passo 2** – Rete neurale scritta da zero con NumPy
- [ ] **Passo 3** – Rete neurale con PyTorch

## Come provarlo

```bash
git clone https://github.com/criskong/italian-town-generator.git
cd italian-town-generator
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

Per rigenerare i dati, scarica l'[elenco dei comuni ISTAT](https://www.istat.it/storage/codici-unita-amministrative/Elenco-comuni-italiani.xlsx),
salvalo in `data/Elenco-comuni-italiani.xlsx` e lancia:

```bash
python prepara_dati.py
```

## Dati

Fonte: ISTAT – Codici delle unità amministrative territoriali.