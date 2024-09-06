# Use an appropriate Python base image
FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy and install dependencies first to leverage Docker caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory
WORKDIR /app

# Copy the rest of the application code
COPY . /app

# Expose the Streamlit default port
EXPOSE 8501

# Set the entrypoint and default command to run Streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]