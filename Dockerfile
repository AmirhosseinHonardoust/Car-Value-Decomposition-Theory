FROM python:3.14-slim

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the rest of the project.
COPY . .

# data/processed, models/, and reports/metrics/ are generated on first run
# by src.train_model (see .gitignore) -- nothing to build here.
EXPOSE 8501

# Uses urllib instead of curl so no extra OS package is needed in the slim image.
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
