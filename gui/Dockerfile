# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Update dependecies
RUN apt-get update

# Set the working directory in the container
WORKDIR /gui

COPY . .

RUN pip install .

RUN pip install -r requirements/requirements.in

RUN python manage.py migrate

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]