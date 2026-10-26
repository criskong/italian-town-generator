# Passo 0 – Preparazione dei dati

## Obiettivo

Trasformare l'elenco ufficiale ISTAT dei comuni italiani (un file Excel con
decine di colonne) in un semplice file di testo con un nome per riga,
pronto per addestrare i modelli.

## Da dove vengono i dati

ISTAT pubblica l'elenco aggiornato di tutti i comuni italiani in formato `.xlsx`.
Dal file prendiamo solo la colonna **"Denominazione in italiano"**.

## Cosa fa lo script `prepara_dati.py`

1. **Apre il file Excel** in sola lettura con la libreria `openpyxl`.
2. **Cerca la riga di intestazione** invece di dare per scontato che sia la prima:
   così lo script regge anche se ISTAT cambia leggermente il formato.
3. **Pulisce i nomi**: toglie spazi in eccesso e mette tutto in minuscolo.
   In questo modo "A" e "a" contano come la stessa lettera e il modello
   ha meno cose da imparare.
4. **Elimina i duplicati** con un `set`: alcuni comuni hanno lo stesso nome
   (ad esempio esistono due Samone e due Peglio).
5. **Salva il risultato** in `data/comuni.txt` e stampa alcune statistiche.

Risultato: **7.889 nomi** unici e un vocabolario di **38 caratteri**
(lettere, lettere accentate, spazio, apostrofo, trattino).

## Concetti chiave

**La qualità dei dati conta più del modello.** Un modello impara quello che
trova nei dati: se i dati sono sporchi (caratteri sbagliati, duplicati),
impara errori.

**Dati grezzi e dati elaborati vanno separati.** Il file Excel originale non è
nel repository: chiunque può riscaricarlo da ISTAT. Nel repository c'è solo
`comuni.txt`, il dato già pulito, insieme allo script che lo produce.
Così il processo è ripetibile.

**Il vocabolario.** È l'elenco di tutti i caratteri che compaiono nei nomi.
Le reti neurali lavorano con numeri, non con lettere: nei passi successivi
ogni carattere del vocabolario riceverà un numero.

## Note di implementazione

- `pathlib.Path` per gestire i percorsi in modo che funzionino su Windows, Mac e Linux
- `openpyxl` per leggere file Excel, con `read_only=True` per risparmiare memoria
- `set` per eliminare i duplicati in automatico
- **type hint** (`-> list[str]`, `int | None`) per documentare cosa restituisce ogni funzione
- `if __name__ == "__main__":` per eseguire il codice solo quando il file viene
  lanciato direttamente, e non quando viene importato da un altro file
- ambiente virtuale (`venv`) e `requirements.txt` per gestire le dipendenze

## Domande frequenti

**Perché tutti i nomi sono in minuscolo?**
Per ridurre il vocabolario: con maiuscole e minuscole separate il modello
dovrebbe imparare due volte le stesse regole, con la metà dei dati per ciascuna.
Le maiuscole vengono rimesse solo al momento di stampare i nomi generati.

**Perché il file Excel originale non è nel repository?**
È un dato grezzo pubblico e scaricabile: nel repository c'è il codice
che lo trasforma e il risultato pulito, non copie di file esterni.