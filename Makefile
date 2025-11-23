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
	uv pip install -r requirements/requirements_dev.txt --system

regenerate-requirements:
	uv pip compile -o requirements/requirements.txt requirements/requirements.in
	uv pip compile -o requirements/requirements_dev.txt requirements/requirements_dev.in

test:
	pytest tests -W ignore::DeprecationWarning


