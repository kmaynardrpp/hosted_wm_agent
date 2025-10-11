# syntax=docker/dockerfile:1

# --------- Stage 1: build the frontend ---------
FROM node:20-alpine AS web
WORKDIR /web
COPY web/infozone-web ./infozone-web
WORKDIR /web/infozone-web
RUN npm ci && npm run build

# --------- Stage 2: Python app ---------
FROM python:3.12-slim

# system libs for matplotlib/reportlab (as you had)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core libjpeg62-turbo libpng16-16 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV INFOZONE_ROOT=/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart

# copy your app code
COPY . ./

# copy the built frontend from stage 1 into the path server.py expects
COPY --from=web /web/infozone-web/dist /app/web/infozone-web/dist

# ensure runtime dirs exist
RUN mkdir -p /app/uploads /app/.runs

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
