FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the rest of the project and install the package itself (deps
# are already satisfied by the layer above) so the `car-value` console
# script works inside the container too, matching the CI quality-gate job.
COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# data/processed, models/, and reports/metrics/ are generated on first run
# by src.train_model (see .gitignore) -- nothing to build here.
EXPOSE 8501

# Uses urllib instead of curl so no extra OS package is needed in the slim image.
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
