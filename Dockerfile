# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Streamlit listens on
EXPOSE 8501

# Streamlit-specific commands to avoid issues
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

# Set the entrypoint to run the Streamlit app
ENTRYPOINT ["streamlit", "run"]

# Set the default command to your application
CMD ["app.py"]
