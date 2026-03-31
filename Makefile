## GUI
.PHONY: build

build:
	cd gui && docker compose build web

build-dev:
	cd gui && docker compose build web web-dev

up:
	- bash -lc 'cd gui && docker compose up db pgadmin redis minio minio-setup web worker --attach web --attach worker'

up-dev: build-dev
	- bash -lc 'cd gui && docker compose up db pgadmin redis minio minio-setup web-dev worker-dev --attach web-dev --attach worker-dev'


down:
	- bash -lc 'cd gui && docker compose down'

## Installation Targets
install-requirements:
	pip install -r requirements/requirements.in

install-requirements-dev:
	pip install -r requirements/requirements_dev.in

install-requirements-uv:
	uv pip install -r requirements/requirements.in

install-requirements-uv-dev:
	uv pip install -r requirements/requirements_dev.in

## CI Targets
regenerate-requirements:
	uv pip compile -o requirements/requirements.txt requirements/requirements.in
	uv pip compile -o requirements/requirements_dev.txt requirements/requirements_dev.in

install-requirements-ci:
	pip install -U uv
	uv pip install -r requirements/requirements_dev.txt --system

test:
	pytest tests -W ignore::DeprecationWarning
