# gui/main/services/method_approval.py

from django.core.exceptions import PermissionDenied
from main.models import QuantumMethod


def require_approved_method(method_name: str) -> QuantumMethod:
    """
    Enforce that a quantum method exists and is approved.

    Single source of truth for approval checks.
    Fail CLOSED: if anything is wrong, deny execution.
    """
    try:
        method = QuantumMethod.objects.get(name=method_name)
    except QuantumMethod.DoesNotExist as exc:
        raise PermissionDenied("ACCESS DENIED: method does not exist.") from exc

    if not method.approved:
        raise PermissionDenied("ACCESS DENIED: method is not approved.")

    return method
