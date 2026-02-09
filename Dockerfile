# Partiamo da un'immagine leggera e ufficiale
FROM python:3.12-slim

# Variabili d'ambiente per ottimizzare Python in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Creiamo una directory di lavoro
WORKDIR /app

# Sicurezza: Aggiorniamo il sistema e installiamo dipendenze minime
# Puliamo la cache di apt per ridurre il peso dell'immagine
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Creiamo un utente di sistema per non eseguire il codice come root
RUN useradd -m appuser
USER appuser

# Copiamo solo il file delle dipendenze per sfruttare la cache dei layer di Docker
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiamo i file necessari (con i permessi corretti)
COPY --chown=appuser:appuser src/api.py .
COPY --chown=appuser:appuser src/scraper.py .
COPY --chown=appuser:appuser outs/ .
# Esponiamo la porta del backend
EXPOSE 8000

# Avviamo l'API tramite lo script che contiene uvicorn.run
CMD ["python", "api.py"]