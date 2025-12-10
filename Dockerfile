FROM python:3.11-slim

# Use /app as the working directory
WORKDIR /app

# Copy dependencies first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest
COPY . .

# Install the orcapy module manually
RUN cd orcapy && python setup.py build
RUN cd orcapy && python setup.py install

# Run your main script
CMD ["python", "auto_run.py"]