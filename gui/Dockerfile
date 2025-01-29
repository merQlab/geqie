# Use an official Python runtime as a parent image
FROM python:3.13-slim

ENV PATH="/usr/local/bin:$PATH"

# Set the working directory in the container
WORKDIR /gui

COPY ./gui .
COPY ./setup.py .
COPY ./geqie .
COPY ./requirements/requirements.in .
COPY ./requirements/requirements_dev.in .

COPY . .

RUN pip install .

EXPOSE 8000

# Command to run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]