# Progetto AC-GAN con Analisi e Grafici

Questo repository contiene il codice per addestrare un Conditional GAN (AC-GAN) e analizzare le sue prestazioni su vari dataset. Inoltre, fornisce uno script per generare grafici che visualizzano le metriche di addestramento e test.

## Contenuti del Repository

1. Codice di Addestramento (`acgan_project.py`):
   - Implementa un Conditional GAN (AC-GAN) utilizzando PyTorch.
   - Supporta diversi dataset come MNIST, FAMNIST, CIFAR10 e CIFAR100.
   - Permette di scegliere la funzione di perdita per l'adversarial loss e l'auxiliary loss.
   - Salva le immagini generate e le metriche di addestramento e test durante l'addestramento.

2. Script per la Generazione di Grafici (`graph.ipynb`):
   - Utilizza Jupyter Notebook per caricare e visualizzare i dati di addestramento e test.
   - Genera grafici delle metriche di addestramento e test, come la perdita del generatore e del discriminatore, l'accuratezza del discriminatore, e le metriche di qualità come l'Inception Score e l'MS-SSIM Score.

3. Requisiti (`requirements.txt`):
   - File che elenca tutte le dipendenze necessarie per eseguire il codice.

## Utilizzo

### 1. Preparazione dell'Ambiente

Assicurati di avere Python 3.6 o superiore installato. Crea un ambiente virtuale (opzionale ma consigliato) e installa le dipendenze richieste:

python -m venv venv
source venv/bin/activate  # Su Windows usa: venv\Scripts\activate
pip install -r requirements.txt

### 2. Esecuzione del Codice di Addestramento

Per addestrare l'AC-GAN, esegui il seguente comando. Puoi personalizzare i parametri di addestramento attraverso gli argomenti della riga di comando:

python train.py --n_epochs 200 --batch_size 64 --lr 0.0002 --b1 0.5 --b2 0.999 --n_cpu 8 --latent_dim 100 --n_classes 10 --img_size 32 --channels 1 --sample_interval 400 --adversarial_loss bce --auxiliary_loss cross_entropy --dataset MNIST --output_dir output

Argomenti principali:
- `--n_epochs`: Numero di epoche di addestramento.
- `--batch_size`: Dimensione del batch.
- `--lr`: Tasso di apprendimento per l'ottimizzatore Adam.
- `--adversarial_loss` e `--auxiliary_loss`: Funzioni di perdita da utilizzare.
- `--dataset`: Dataset da utilizzare (MNIST, CIFAR10, ecc.).
- `--output_dir`: Directory per salvare le immagini e i risultati.

### 3. Generazione di Grafici

Dopo l'addestramento, puoi visualizzare i grafici delle metriche eseguendo lo script Jupyter Notebook incluso. Assicurati di avere Jupyter Notebook installato.

jupyter notebook graph.ipynb

Segui le istruzioni nel notebook per caricare i dati generati dall'addestramento e visualizzare i grafici delle metriche come la perdita del generatore e del discriminatore, l'accuratezza del discriminatore, l'Inception Score e l'MS-SSIM Score.

## Contatti

Per qualsiasi domanda o problema, puoi contattare Maria Grazia Borzì a m.borzi@outlook.com.
