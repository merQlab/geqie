import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Don't fail while geqie-qml has no tests yet.
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
