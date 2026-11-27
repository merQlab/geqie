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
	uv pip install -r requirements/requirements.in

install-requirements-dev:
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
		geqie/requirements/requirements_dev.txt \
		geqie-qml/requirements/requirements.in \
		geqie-qml/requirements/requirements_dev.in \
		-o geqie-qml/requirements/requirements_dev.txt

# Constrained by the geqie dev lock so both can be installed into one environment.
regenerate-requirements-docs:
	uv pip compile \
		docs/requirements/requirements.in \
		-c geqie/requirements/requirements_dev.txt \
		-o docs/requirements/requirements.txt

regenerate-requirements: regenerate-requirements-geqie regenerate-requirements-geqie-qml regenerate-requirements-docs


## Documentation
install-requirements-docs:
	uv pip install -r docs/requirements/requirements.txt

docs-serve:
	zensical serve

docs-build:
	zensical build --clean --strict


## Tests
test-geqie:
	pytest geqie/tests -W ignore::DeprecationWarning

test-docs:
	pytest \
		README.md \
		geqie/README.md \
		geqie-qml/README.md \
		docs \
		--markdown-docs --markdown-docs-syntax=superfences -W ignore::DeprecationWarning

# Exit code 5 means "no tests collected"; tolerate it while geqie-qml has no tests.
test-geqie-qml:
	pytest geqie-qml/tests -W ignore::DeprecationWarning

test: test-geqie test-geqie-qml test-docs


## CI Setup
install-requirements-ci:
	pip install -U uv
	uv pip install --system \
		"./geqie[dev]" \
		-r docs/requirements/requirements.txt
# 	uv pip install ./geqie-qml --system
