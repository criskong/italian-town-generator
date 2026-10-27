# Passo 3 – Rete neurale con PyTorch

## Obiettivo

Superare il limite dei passi precedenti, in cui il modello vedeva una sola lettera.
La nuova rete guarda le **ultime 3 lettere**, rappresenta ogni lettera con un
**embedding** imparato e ha uno **strato nascosto** di neuroni.
Viene inoltre valutata su nomi che non ha mai visto.

## Le novità

### 1. Più contesto

Per indovinare la prossima lettera la rete guarda le 3 precedenti. Con "napoli":

| Contesto | Lettera da indovinare |
|---|---|
| `. . .` | n |
| `. . n` | a |
| `. n a` | p |
| `n a p` | o |
| `a p o` | l |
| `p o l` | i |
| `o l i` | `.` |

Con il conteggio questo non sarebbe praticabile: servirebbe una tabella di
39⁴ ≈ 2,3 milioni di celle, quasi tutte vuote con soli 95.000 esempi.
La rete invece **generalizza**: quello che impara su combinazioni frequenti
vale anche per combinazioni mai viste.

### 2. Gli embedding

Nel Passo 2 ogni lettera era un vettore one-hot: per la rete la `a` era diversa
dalla `e` quanto dalla `z`. Ora ogni lettera è rappresentata da **10 numeri
che la rete impara** durante l'addestramento. Lettere che si comportano in modo
simile ottengono numeri simili, così quello che la rete impara su una lettera
aiuta anche con quelle "vicine".

### 3. Lo strato nascosto

Tra ingresso e uscita c'è uno strato di **200 neuroni** con attivazione `tanh`.

```
3 lettere → embedding (3 × 10 = 30 numeri) → 200 neuroni (tanh) → 39 logits
```

La `tanh` è una funzione **non lineare**: senza di essa, due strati lineari
di fila equivarrebbero matematicamente a uno solo, e la rete non potrebbe
imparare combinazioni complesse tra le lettere.

In totale la rete ha **14.429 parametri**.

### 4. Training, validation e test

Una rete con molti parametri può imparare a memoria i dati invece di capirne
le regole. Per accorgersene, i nomi sono divisi in tre gruppi:

| Gruppo | Nomi | Uso |
|---|---|---|
| Training | 6.311 (80%) | la rete impara solo da questi |
| Validation | 789 (10%) | controllo durante l'addestramento |
| Test | 789 (10%) | valutazione finale, guardato una sola volta |

La divisione è fatta **per nome**, non per coppie di lettere: così nessun pezzo
di un nome di validation o test finisce nei dati di addestramento.

## Cosa fa PyTorch

Il ciclo di addestramento è lo stesso del Passo 2: forward, backward, update.
Cambia chi fa i conti.

| Passo 2 (a mano) | Passo 3 (PyTorch) |
|---|---|
| `X @ W` | `nn.Linear` |
| `softmax` + loss | `F.cross_entropy` |
| calcolo manuale di `dW` | `loss.backward()` |
| `W -= learning_rate * dW` | `ottimizzatore.step()` |

- **Autograd** (`loss.backward()`): PyTorch registra le operazioni del forward
  e calcola automaticamente i gradienti di tutti i parametri, per qualsiasi numero di strati.
- **`zero_grad()`**: i gradienti si sommano a ogni `backward()`, quindi vanno
  azzerati prima di ogni passo.
- **Minibatch**: ogni passo usa 32 esempi casuali invece di tutti i 76.099.
  I passi sono meno precisi ma molto più veloci.
- **Learning rate decrescente**: 0,1 per i primi 20.000 passi, poi 0,01 per
  sistemarsi meglio sul minimo.

## I risultati

### La curva di addestramento

| Passo | Training | Validation |
|---|---|---|
| 0 | 3,677 | 3,676 |
| 2.000 | 2,117 | 2,126 |
| 10.000 | 1,918 | 1,955 |
| 20.000 | 1,868 | 1,939 |
| 22.000 | 1,791 | 1,856 |
| finale | 1,777 | 1,850 |

- Training e validation scendono insieme: la rete impara regole generali.
- La distanza tra le due curve cresce lentamente (da 0,008 a 0,07):
  un leggero overfitting, sotto controllo.
- Tra i passi 12.000 e 20.000 la validation si blocca: con learning rate 0,1
  i passi sono troppo lunghi per scendere ancora.
- Al passo 22.000, dopo il passaggio a learning rate 0,01, la loss scende di colpo.

### Confronto con i modelli precedenti

| Modello | Loss | Perplessità |
|---|---|---|
| Casuale | 3,664 | 39,0 |
| Bigrammi | 2,260 | 9,6 |
| Rete NumPy | 2,268 | 9,7 |
| **Rete PyTorch (test set)** | **1,870** | **6,5** |

La loss della rete PyTorch è misurata su nomi **mai visti**, quella dei bigrammi
sugli stessi nomi usati per contare: il miglioramento reale è ancora maggiore
di quanto suggeriscano i numeri.

### I nomi generati

- Credibili: *Pava Lucia*, *Monteto*, *Civero*, *Sarsi*, *Pulagonforgia*
- Con struttura da nome composto: *Fossa Di Briano Di Novalles*, *Bulla San Barenti*
- Un solo nome su 20 coincide con un comune reale (*Agnone*): la rete inventa, non ricopia.

### Limiti

Con 3 lettere di memoria la rete non sa quanto è già lungo il nome, quindi a volte
genera nomi molto lunghi come *Rivanucciaverre In Rocuzze Conchiemaggiaco*.
Per migliorare servirebbe più contesto o un'architettura con memoria più lunga
(reti ricorrenti o Transformer).

## Note di implementazione

- La rete è una classe che eredita da `nn.Module`: i parametri dei suoi strati
  vengono registrati automaticamente e passati all'ottimizzatore con `modello.parameters()`.
- `emb.view(emb.shape[0], -1)` unisce gli embedding delle 3 lettere in un unico
  vettore da 30 numeri senza copiare i dati.
- `@torch.no_grad()` disattiva l'autograd durante valutazione e generazione:
  più veloce e meno memoria.
- `torch.manual_seed` e `torch.Generator` rendono riproducibili sia i pesi iniziali
  sia la scelta dei minibatch e la generazione.

## Domande frequenti

**Cos'è l'overfitting e come si riconosce?**
È quando la rete memorizza i dati di addestramento invece di impararne le regole.
Si riconosce quando la loss di training continua a scendere mentre quella di
validation si ferma o risale.

**Perché tre gruppi di dati e non due?**
La validation serve a prendere decisioni (quando fermarsi, che parametri usare).
Proprio per questo non è più imparziale: il test, guardato una sola volta alla fine,
dà la misura onesta delle prestazioni.

**Perché usare minibatch invece di tutti i dati?**
Ogni passo costa molto meno. Tanti passi approssimativi portano più lontano,
nello stesso tempo, di pochi passi precisi.

**Che vantaggio danno gli embedding rispetto al one-hot?**
Sono più compatti e, soprattutto, imparati: lettere con comportamento simile
ottengono rappresentazioni simili, e la rete può trasferire quello che impara
da una lettera all'altra.