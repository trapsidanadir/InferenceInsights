FROM python:3.11-slim
WORKDIR /app
COPY ./. /app/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Set environment variables to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install curl
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*