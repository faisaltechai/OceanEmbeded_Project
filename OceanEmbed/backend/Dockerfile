FROM python:3.12-slim

WORKDIR /srv
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY ml ./ml
COPY data/demo ./data/demo

ENV OCEANEMBED_BUNDLE_PATH=/srv/data/demo/backend_bundle.json
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
