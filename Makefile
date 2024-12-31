install-requirements:
	pip install -r requirements/requirements.in

install-requirements-dev:
	pip install -r requirements/requirements_dev.in

install-requirements-uv:
	uv pip install -r requirements/requirements.in

install-requirements-dev-uv:
	uv pip install -r requirements/requirements_dev.in


test:
	pytest tests -W ignore::DeprecationWarning