FROM python:3.11-slim

# Use /app as the working directory
WORKDIR /app

# Copy dependencies first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest
COPY . .

# Run your main script
CMD ["python", "auto_run.py"]