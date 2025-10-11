# syntax=docker/dockerfile:1
FROM python:3.12-slim

# System libs helpful for matplotlib/reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core libjpeg62-turbo libpng16-16 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV INFOZONE_ROOT=/app

# deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart

# app code + built frontend
COPY . ./
RUN mkdir -p /app/uploads /app/.runs

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
