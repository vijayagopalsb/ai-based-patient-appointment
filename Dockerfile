# Use an official Python runtime as a base image

FROM python:3.12

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory in the container
WORKDIR /app

# Copy project files to the container
COPY . /app/


# Install dependencies
RUN pip install --upgrade pip \
    && pip install --root-user-action=ignore -r requirements.txt

# Default command to run your application
ENTRYPOINT ["python", "main_controller.py"]

