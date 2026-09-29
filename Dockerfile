# ==============================================================================
# Dockerfile for Food Adulteration Risk Prediction System
# Academic Submission: K. Ramakrishnan College of Technology (Autonomous), Trichy
# ==============================================================================

FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy pinned requirements specification
COPY requirements.txt .

# Install Python dependencies (ignoring cache to reduce image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete project code and pre-trained model artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY db/ ./db/
COPY reports/ ./reports/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose standard Streamlit port
EXPOSE 8501

# Healthcheck to ensure container is serving web traffic
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit dashboard
ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
