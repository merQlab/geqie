# Running on IBM Quantum

[`geqie.execute`][geqie.main.execute] transpiles the circuit for an IBM Quantum Platform backend and runs it with the Qiskit Runtime `SamplerV2` primitive.

## Credentials

By default, the account saved with [`QiskitRuntimeService.save_account`](https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/save-credentials) is used:

```{.python notest}
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(token=API_TOKEN, instance=INSTANCE_CRN, set_as_default=True)  # once

counts = geqie.execute(circuit, n_shots=1024)
```

To use a specific saved account, pass its name:

```{.python notest}
counts = geqie.execute(circuit, n_shots=1024, credentials_name="my-ibm-credentials")
```

Alternatively, authenticate with an API token and instance CRN directly:

```{.python notest}
counts = geqie.execute(
    circuit,
    n_shots=1024,
    api_token=API_TOKEN,
    instance_crn=INSTANCE_CRN,
)
```

The same options are available in the [CLI](cli.md#geqie-execute).

!!! warning
    Never commit API tokens to source control. Load them from environment variables or use saved credentials.

## Backend selection

Pass `backend="ibm_fez"` to target a specific backend. If omitted, the least busy operational backend with enough qubits for the circuit is selected.

## Transpilation and dry runs

Extra arguments for `qiskit.transpile` are passed via `transpiler_args`. Set `dry_run=True` to set up the backend and transpile the circuit without submitting the job:

```{.python notest}
geqie.execute(
    circuit,
    n_shots=1024,
    credentials_name="my-ibm-credentials",
    transpiler_args={"optimization_level": 3},
    dry_run=True,
    logging_level="DEBUG",
)
```

## Results

By default, `geqie.execute` returns measurement counts that can be passed directly to [`geqie.retrieve`][geqie.main.retrieve]. Set `return_qiskit_result=True` to get the raw Qiskit Runtime result instead.

See [`execute.ipynb`](https://github.com/merQlab/geqie/blob/main/examples/execute.ipynb) for a complete example.
