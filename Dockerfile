# Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    gcc \
    g++ \
    make \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib dependencies
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the project as a package
COPY setup.py .
RUN pip install -e .

# Copy project files, keeping only the essential directories
COPY ai_trading /app/ai_trading
COPY web_app /app/web_app
COPY tradingview /app/tradingview
COPY data /app/data
COPY tests /app/tests

# Create necessary directories
RUN mkdir -p data

# Expose port for web app
EXPOSE 8000

# Choose one of the following commands based on your needs:

# Run tests
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Run the web application
# CMD ["python", "-m", "web_app.app"]

# Run a training session
# CMD ["python", "-m", "ai_trading.train", "--download", "--symbol", "BTC/USDT", "--timeframe", "1h", "--days", "60", "--backtest"] 