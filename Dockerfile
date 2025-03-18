# Use official Python image as base
FROM python:3.10

# Set working directory in the container
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 8080

# Command to run the app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]