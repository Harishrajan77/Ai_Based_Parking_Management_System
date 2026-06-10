# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Set environment variable for YOLO model path so it looks in the current directory
ENV MODEL_PATH=/app/best.pt

# Expose the port the app runs on
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py"]