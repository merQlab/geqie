install-requirements:
	pip install -r requirements/requirements.in

install-requirements-dev:
	pip install -r requirements/requirements_dev.in

install-requirements-uv:
	uv pip install -r requirements/requirements.in

install-requirements-uv-dev:
	uv pip install -r requirements/requirements_dev.in


install-requirements-ci:
	pip install -U uv
	uv pip install -r requirements/requirements_dev.in --system

test:
	pytest tests -W ignore::DeprecationWarning


