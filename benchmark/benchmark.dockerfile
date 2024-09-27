FROM python:3.11
WORKDIR /app
COPY ./. /app/
RUN pip install --no-cache-dir -r requirements.txt

#Docker CLI
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh