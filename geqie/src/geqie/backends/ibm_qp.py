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

    if credentials_name == "" and api_token == "":
        raise ValueError("Either 'credentials_name' or 'api_token' + 'instance_name' must be provided.")
    
    if credentials_name != "":
        ibm_qp_service = QiskitRuntimeService(
            credentials_name, 
            channel=channel,
            **iqp_runtime_args,
        )
    else:
        ibm_qp_service = QiskitRuntimeService(
            channel=channel,
            token=api_token,
            instance=instance_crn,
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
