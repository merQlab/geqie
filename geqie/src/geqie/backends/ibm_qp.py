from typing import Any, Dict

from qiskit_ibm_runtime import SamplerV2 as Sampler, QiskitRuntimeService


def get_ibm_quantum_backend(
        api_token: str = "",
        instance_crn: str = "",
        credentials_name: str = "",
        backend: str = "",
        channel: str = "ibm_quantum_platform",
        iqp_runtime_args: Dict[str, Any] = {},
        min_num_qubits: int | None = None,
) -> Any:
    ibm_qp_service = QiskitRuntimeService(
        # A named account determines its own channel.
        channel=None if credentials_name else channel,
        token=api_token or None,
        instance=instance_crn or None,
        name=credentials_name or None,
        **iqp_runtime_args,
    )
    
    if backend != "":
        return ibm_qp_service.backend(backend)
    else:
        return ibm_qp_service.least_busy(
            simulator=False, 
            operational=True, 
            min_num_qubits=min_num_qubits,
        )


def get_sampler(backend: Any) -> Sampler:
    return Sampler(mode=backend)
