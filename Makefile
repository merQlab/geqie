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


## Regenerate requirements
regenerate-requirements-geqie:
	uv pip compile \
		geqie/requirements/requirements.in \
		-o geqie/requirements/requirements.txt
	uv pip compile \
		geqie/requirements/requirements.in \
		geqie/requirements/requirements_dev.in \
		-o geqie/requirements/requirements_dev.txt

regenerate-requirements-geqie-qml:
	uv pip compile \
		geqie/requirements/requirements.txt \
		geqie-qml/requirements/requirements.in \
		-o geqie-qml/requirements/requirements.txt
	uv pip compile \
		geqie/requirements/requirements.txt \
		geqie-qml/requirements/requirements.in \
		geqie-qml/requirements/requirements_dev.in \
		-o geqie-qml/requirements/requirements_dev.txt

regenerate-requirements: regenerate-requirements-geqie regenerate-requirements-geqie-qml


## CI Targets

install-requirements-ci:
	pip install -U uv
	uv pip install ./geqie[dev] --system
# 	uv pip install ./geqie-qml --system

test:
	pytest tests -W ignore::DeprecationWarning
