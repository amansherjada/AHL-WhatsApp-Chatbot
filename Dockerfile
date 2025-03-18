# Use the official Python image with reduced size for better performance
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first (to leverage Docker caching)
COPY requirements.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Ensure proper execution permissions
RUN chmod +x app_with_firebase.py

# Set the entrypoint command
CMD ["python", "app_with_firebase.py"]
