# Passo 2 – Rete neurale scritta da zero con NumPy

## L'idea

Costruire una rete neurale che fa **lo stesso lavoro del modello a bigrammi**:
riceve una lettera e predice le probabilità della successiva.

La differenza sta nel modo in cui si ottengono i numeri:

- nei **bigrammi** le probabilità si calcolano **contando**;
- nella **rete** si parte da numeri **casuali** e li si corregge un po' alla volta
  per abbassare la loss.

L'obiettivo del passo è verificare che la rete, senza che nessuno le dica
di contare, arrivi da sola agli stessi risultati.

## Il dataset

I nomi vengono trasformati in coppie (lettera attuale → lettera successiva):

```
napoli → (.→n) (n→a) (a→p) (p→o) (o→l) (l→i) (i→.)
```

In totale sono **94.931 coppie**. La lettera attuale è l'**ingresso**,
quella successiva è la **risposta giusta**.

### La codifica one-hot

Ogni lettera viene trasformata in un vettore di 39 posizioni con un solo 1.
Con 5 simboli, il simbolo numero 2 diventerebbe `[0, 0, 1, 0, 0]`.

Non si passa direttamente il numero della lettera perché la rete lo tratterebbe
come una quantità: la lettera 14 sembrerebbe "più grande" della 13,
e non ha senso. Con il one-hot tutte le lettere sono alla pari.

## La rete

La rete è una sola tabella di pesi `W` di dimensione 39 × 39.
**È tutta la conoscenza del modello**: all'inizio contiene numeri casuali,
alla fine contiene quello che ha imparato.

## Il ciclo di addestramento

La rete impara ripetendo 300 volte tre operazioni.

### 1. Forward: dalla lettera alle probabilità

```
logits = X @ W
probabilità = softmax(logits)
```

- `X @ W` moltiplica le lettere one-hot per i pesi. Con il one-hot, questo
  equivale a **prendere la riga di `W` della lettera di ingresso**.
- Il risultato (i **logits**) sono numeri qualsiasi, anche negativi.
- La **softmax** li trasforma in probabilità: calcola `exp()` di ogni numero
  (così diventano positivi) e divide per la somma (così sommano a 1).

Poi si calcola la **loss**, la stessa del Passo 1: la sorpresa media
davanti alle lettere vere.

### 2. Backward: in che direzione correggere i pesi

Per ogni esempio, la correzione sui logits è:

```
probabilità predette − risposta giusta (one-hot)
```

Esempio: dopo la `q` la rete dava il 30% alla `u`, che era la risposta giusta.
La correzione è 0,30 − 1 = **−0,70**: il peso della `u` va alzato molto.
Per una lettera sbagliata a cui dava il 10%, la correzione è 0,10 − 0 = **+0,10**:
quel peso va abbassato un po'.

**Più la rete sbaglia, più forte è la correzione.** Questo vettore di correzioni
si chiama **gradiente**, e il procedimento per calcolarlo **backpropagation**.

Con `dW = X.T @ dlogits` le correzioni passano dai logits ai pesi:
grazie al one-hot, gli errori fatti dopo una certa lettera
correggono solo la riga di quella lettera.

### 3. Update: un piccolo passo in discesa

```
W -= learning_rate * dW
```

Ogni peso viene spostato di poco nella direzione che abbassa la loss.
Il metodo si chiama **discesa del gradiente**: come scendere una collina
nella nebbia, facendo ogni volta un passo dove il terreno scende.

Il **learning rate** è la lunghezza del passo:

- troppo piccolo → la rete impara lentissimamente;
- troppo grande → "salta" oltre la valle e la loss oscilla o esplode.

## I risultati

### La curva della loss

| Passo | Loss |
|---|---|
| 0 | 4,100 |
| 25 | 2,416 |
| 100 | 2,294 |
| 200 | 2,274 |
| finale | 2,268 |

- All'inizio la loss è **peggiore del modello casuale** (3,664): i pesi casuali
  rendono la rete sicura di previsioni sbagliate, e la loss punisce proprio questo.
- Il grosso dell'apprendimento avviene nei primi passi, poi la discesa rallenta.
- Il valore finale è praticamente uguale a quello dei bigrammi (2,260).

### La rete ha riscoperto i conteggi

| Lettera iniziale | Bigrammi (contati) | Rete (imparati) |
|---|---|---|
| c | 18,5% | 18,5% |
| s | 13,7% | 13,7% |
| m | 10,5% | 10,4% |

Altre regole imparate dagli esempi:

- dopo la `q`, la `u` ha il **91%** di probabilità;
- dopo la `z`, un'altra `z` ha il **31%**: è la doppia z di nomi come Pozzuoli o Arzano.

### Perché non fa meglio dei bigrammi

La rete vede **una sola lettera**, esattamente come i bigrammi: ha la stessa
informazione, quindi non può fare previsioni migliori. Il limite non è
l'addestramento ma il contesto. I nomi generati hanno infatti la stessa qualità:
alcuni credibili (*Gintenda Bulona*, *Ma Gioro*), altri no (*Trrcasano*).

I caratteri rarissimi (lettere accentate) compaiono in pochi esempi, quindi
le loro righe di `W` ricevono poche correzioni e restano vicine ai valori casuali
iniziali. Dopo una di queste lettere la rete sceglie quasi a caso,
come succedeva con lo smoothing nei bigrammi.

## Note di implementazione

- Tutti i calcoli sono **vettorizzati** con NumPy: si elaborano le 94.931 coppie
  in una sola operazione, senza cicli `for` in Python.
- `probabilita[np.arange(n), Y]` (fancy indexing) estrae, per ogni riga,
  la probabilità della lettera giusta.
- `keepdims=True` nella softmax mantiene la forma della matrice, così la divisione
  avviene riga per riga grazie al **broadcasting**.
- Nella softmax si sottrae il massimo di ogni riga prima di `exp()`: il risultato
  non cambia, ma si evita l'overflow con numeri grandi.
- `np.random.default_rng(seme)` rende l'addestramento riproducibile.

## Domande frequenti

**Se il risultato è uguale ai bigrammi, a cosa serve la rete?**
Il conteggio non si può estendere: guardare più lettere indietro fa esplodere
la dimensione della tabella. La rete invece si può ingrandire
(più contesto, più strati) mantenendo lo stesso metodo di addestramento.

**Perché la loss iniziale è peggiore di quella del modello casuale?**
Il modello casuale dà la stessa probabilità a ogni lettera. I pesi casuali invece
creano preferenze arbitrarie: la rete è "sicura" di lettere sbagliate,
e la loss punisce molto la sicurezza mal riposta.

**Cosa succede con un learning rate sbagliato?**
Troppo piccolo: l'addestramento è lentissimo. Troppo grande: i passi saltano
oltre il minimo e la loss oscilla o diverge.