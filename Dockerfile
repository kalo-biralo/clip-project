# Use an official Python runtime as a parent image
FROM python:3.10-slim

RUN pip install poetry

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN poetry install

# Expose the port streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "main.py"]
