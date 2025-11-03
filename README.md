# 🧠 Mental Support Chatbot

Chatbot dedicato al supporto della salute mentale e al cambiamento comportamentale, finalizzato al raccoglimento di informazioni utili alla definizione del profilo psicologico dell'utente.  

## 🚀 Sviluppo
**Tech used**: Rasa Open Source, Python, CSS, HTML  

- **Rasa** è un framework open-source per la costruzione di chatbot e assistenti virtuali basati su machine learning.  
  Fornisce strumenti per Natural Language Understanding (NLU), gestione del dialogo (Dialogue Management) e integrazione con modelli personalizzati.  
  Permette di creare sistemi conversazionali scalabili, personalizzabili e in grado di gestire conversazioni complesse, in particolare tramite l'integrazione con **Python** per la gestione delle azioni personalizzate.  

- **HTML e CSS** sono stati utilizzati per sviluppare l'interfaccia di base per l'utilizzo del chatbot.  

## 📦 Requisiti
- **Python 3.10.11**  
- **Rasa Open Source** (installabile con `pip install rasa`)  
- Ambiente virtuale Python (`venv`)
- **LM Studio** (per l’esecuzione di modelli locali di linguaggio)  
- **Mistral-7B-Instruct v0.3** (modello locale LLM utilizzato dal chatbot)

  
## 📂 Struttura del progetto
La struttura principale del progetto è la seguente:
```plaintext
├── actions/          # Azioni personalizzate in Python  
├── data/             # Training data: nlu.yml, stories.yml, rules.yml  
├── models/           # Modelli addestrati salvati da Rasa  
├── domain.yml        # Definizione di intent, entità, slot e risposte  
├── config.yml        # Configurazione pipeline NLU e politica di dialogo  
├── credentials.yml   # Configurazioni per canali esterni  
├── endpoints.yml     # Endpoint per azioni personalizzate e tracker store  
├── frontend/         # Interfaccia utente (index.html, CSS, JS)  
└── requirements.txt  # Dipendenze del progetto
```  
## ▶️ Come utilizzare
Clona il repository e spostati nella directory del progetto. Crea ed attiva l’ambiente virtuale:  
```bash
py -3.10 -m venv venv  
venv\Scripts\activate   # Windows  
source venv/bin/activate # Linux/Mac
```

Installa le dipendenze:
```bash
pip install -r requirements.txt
```
Addestra il modello ed esegui il server Rasa:
```bash
rasa train
rasa run -m models --enable-api --cors "*" --debug
```
In un altro terminale, avvia le azioni personalizzate:
```bash
venv\Scripts\activate   # (o source venv/bin/activate su Linux/Mac)
rasa run actions
```

Apri il file index.html all’interno della cartella frontend/ nel browser per interagire con il chatbot.
