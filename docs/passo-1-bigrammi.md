# Passo 1 – Modello a bigrammi

## L'idea

Un **bigramma** è una coppia di lettere consecutive. Prendiamo il nome
"napoli" e aggiungiamo un segnale `.` all'inizio e alla fine:

```
. n a p o l i .
```

Le coppie sono: `.n` `na` `ap` `po` `ol` `li` `i.`

Il `.` iniziale permette di imparare **come iniziano** i nomi,
quello finale **quando finiscono**.

## L'addestramento: contare

Il modello legge tutti i 7.889 comuni e conta quante volte ogni lettera
è seguita da ogni altra lettera. Il risultato è una tabella 39 × 39:
la riga della `q`, per esempio, dice che dopo la `q` arriva quasi sempre la `u`.

Già da qui il modello "scopre" cose vere sull'italiano. Le lettere
iniziali più frequenti sono:

| Lettera | Frequenza | Perché |
|---|---|---|
| c | 18,5% | Castel-, Casal-, Campo-, Civita- |
| s | 13,7% | San, Santa, Sant' |
| m | 10,5% | Monte-, Mont- |

## Da conteggi a probabilità

Ogni riga della tabella viene trasformata in probabilità che sommano a 1.
Se dopo la `q` la `u` compare 95 volte su 100, la sua probabilità è il 95%.

### Lo smoothing

Prima di calcolare le probabilità si aggiunge **+1 a ogni conteggio**.
Senza questo trucco, una coppia mai vista avrebbe probabilità zero
e il modello la considererebbe impossibile.

Lo smoothing ha però un effetto collaterale: per i caratteri rarissimi
(come le lettere accentate dei comuni della Valle d'Aosta) i conteggi reali
sono pochissimi, quindi il +1 pesa più dei dati veri. Dopo un carattere raro
il modello sceglie quasi a caso: ecco perché è uscito un nome come *Éçôòzanal*.

## La generazione: il campionamento

Per inventare un nome:

1. si parte dal segnale `.`
2. si estrae la lettera successiva **a caso, ma rispettando le probabilità**
3. si ripete con la nuova lettera
4. ci si ferma quando esce di nuovo `.`

È lo stesso meccanismo con cui i grandi modelli linguistici scelgono
la parola successiva, solo molto più semplice.

## La loss: misurare quanto ha imparato il modello

### L'intuizione

Immagina un gioco: si scrive un nome una lettera alla volta e il modello
deve indovinare la prossima.

- Un **modello a caso** non sa niente: per lui tutti i 39 simboli sono
  ugualmente probabili. È come se a ogni lettera fosse indeciso tra 39 scelte.
- Il **modello a bigrammi** ha imparato dai dati: è come se fosse indeciso
  tra circa 10 scelte.

La **loss** misura proprio questa confusione. **Più è bassa, meglio è.**

### Come si calcola

Per ogni coppia di un nome vero si guarda che probabilità le dava il modello,
e la si trasforma in "sorpresa" con `-log(p)`:

| Probabilità data alla lettera giusta | Sorpresa |
|---|---|
| 90% | 0,105 |
| 50% | 0,693 |
| 10% | 2,303 |
| 1% | 4,605 |

La loss è la **sorpresa media** su tutte le coppie di tutti i nomi.
Il nome tecnico è *negative log-likelihood*.

### I risultati

| Modello | Loss | Perplessità |
|---|---|---|
| Casuale | 3,664 | 39,0 |
| Bigrammi | 2,260 | 9,6 |

La **perplessità** si calcola come `e^loss` ed è il numero di scelte
tra cui il modello è indeciso in media. Il modello a bigrammi riduce
l'incertezza da 39 a circa 10 scelte per lettera.

## Risultati e limiti

Alcuni nomi generati sono credibili: *Cilano*, *Miascano*, *Velno*, *Sgog D'Oro*.

Altri mostrano i limiti del modello:

- *Brlonnzilololl*: il modello ricorda **una sola lettera**, quindi non sa
  di stare ripetendo la stessa sillaba.
- *A*, *So*, *Oco*: non sa quanto è già lungo il nome, quindi può fermarsi troppo presto.

Per fare meglio serve un modello che guardi **più lettere indietro**:
è l'obiettivo dei passi successivi.

## Cosa ho imparato in Python

- `zip(lettere, lettere[1:])` per accoppiare ogni elemento con il successivo
- `[[0] * n for _ in range(n)]` per creare una tabella. Attenzione:
  `[[0] * n] * n` crea n riferimenti **alla stessa riga**, quindi modificandone
  una si modificano tutte
- `random.Random(seme)` per avere risultati riproducibili
- `rng.choices(..., weights=...)` per estrarre a caso con probabilità diverse
- `from bigrammi import ...` per riusare funzioni da un altro file

## Possibili domande da colloquio

**Cos'è un modello generativo?**
Un modello che impara la distribuzione dei dati e può produrne di nuovi
dello stesso tipo. Qui impara come sono fatti i nomi dei comuni e ne inventa di nuovi.

**Cos'è la loss e perché si usa il logaritmo?**
Misura quanto il modello è "sorpreso" dai dati veri. Il logaritmo trasforma
il prodotto di tante probabilità piccole in una somma, e penalizza molto
le previsioni sbagliate fatte con sicurezza.

**Perché serve un modello di riferimento casuale?**
Senza un termine di paragone non si sa se una loss è buona o cattiva.
Se il modello non batte il caso, non ha imparato niente.

**Qual è il limite principale dei bigrammi?**
Hanno memoria di una sola lettera. Allungare la memoria contando
(trigrammi, quadrigrammi...) fa esplodere la dimensione della tabella:
è uno dei motivi per cui si passa alle reti neurali.