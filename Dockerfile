FROM python:3.10-slim

ENV POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN pip install "poetry>=2,<3"

WORKDIR /app

# Install dependencies first so this layer is cached until they change
COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
RUN poetry install --only main

# Trained weights are not baked into the image; mount them at run time:
#   docker run -p 8501:8501 -v "$(pwd)/weights:/app/weights" clip-demo
ENV CLIP_CHECKPOINT=/app/weights/best_checkpoint.pth

EXPOSE 8501
CMD ["streamlit", "run", "src/clip/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
