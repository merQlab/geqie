const methodSelect = document.getElementById('methodSelect');
const initContent = document.getElementById('initContent');
const mapContent = document.getElementById('mapContent');
const dataContent = document.getElementById('dataContent');

document.addEventListener("DOMContentLoaded", function () {
    methodSelect.addEventListener("change", function () {
        const methodName = methodSelect.value;
        if (!methodName) {
            initContent.value = "";
            mapContent.value = "";
            dataContent.value = "";
            return;
        }

        fetch(`/get-method/${methodName}/`)
            .then(response => response.json())
            .then(data => {
                initContent.value = data.init || "No content found.";
                mapContent.value = data.map || "No content found.";
                dataContent.value = data.data || "No content found.";
            })
            .catch(error => logToServer('critical', 'Error fetching method data: ${error}'));
    });
});

document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("addInitContent").value = 
    `import numpy as np
from qiskit.quantum_info import Statevector

def init(n_qubits: int) -> Statevector:    
    qubits_in_superposition = # place number of qubits in superposition
    base_state = np.zeros(2**qubits_in_superposition , dtype=int)    
    base_state[0] = 1    
    state = np.tile(base_state, 2**(n_qubits - qubits_in_superposition ))    
    return Statevector(state)`;

    document.getElementById("addMapContent").value = 
    `import numpy as np
from qiskit.quantum_info import Operator

def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:    
    p = image[u, v]  
    # Provide your own unitary matrix for map operator
    return Operator(map_operator)`;

    document.getElementById("addDataContent").value = 
    `import numpy as np
from qiskit.quantum_info import Statevector

def data(u: int, v: int, R: int, image: np.ndarray) -> Statevector:    
    m = u * image.shape[0] + v    
    data_vector = np.zeros(2**(2 * R))    
    data_vector[m] = 1    
    return Statevector(data_vector)`;
});

let formData = new FormData();

document.getElementById("imagePath").addEventListener("click", function() {
    document.getElementById("fileInput").click();
});

document.getElementById("fileInput").addEventListener("change", function() {
    const filePathInput = document.getElementById("filePath");
    if (this.files && this.files[0]) {
        filePathInput.value = this.files[0].name;
    }
});

document.getElementById("addNewMethod").addEventListener("click", function () {
    logToServer('info', 'Method added.');
    saveMethod(true, true, document.getElementById("methodName").value, "methodName", "addInitContent", "addMapContent", "addDataContent");
});

document.getElementById("save").addEventListener("click", function () {
    logToServer('info', 'Method saved.');
    saveMethod(false, false, document.getElementById("methodSelect").value, "methodSelect", "initContent", "mapContent", "dataContent");
});

document.getElementById("saveAsNew").addEventListener("click", function () {
    let userInput = prompt("Method name:");
    if (userInput === null) {
        alert("Canceled");
    } else if (userInput.trim() === "") {
        alert("Name is empty");
    } else {
        logToServer('info', 'Method saved as new.');
        saveMethod(false, true, userInput.trim(), "methodSelect", "initContent", "mapContent", "dataContent");
    }
});

async function saveMethod(addNew, isNew, saveName, method, init, map, data) {
    let methodName = document.getElementById(method).value;
    let initContent = document.getElementById(init).value;
    let mapContent = document.getElementById(map).value;
    let dataContent = document.getElementById(data).value;

    if (!methodName) {
        alert("Please select a method first.");
        return;
    }

    const canProceed = await checkFolderExists(methodName);
    if (!canProceed) {
        return;
    }

    fetch("/save-method/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken(),
        },
        body: JSON.stringify({
            method_name: methodName,
            init: initContent,
            map: mapContent,
            data: dataContent,
            is_new: isNew,
            save_name: saveName,
            add_new: addNew,
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            logToServer('warning', 'Error: ${data.error}')
            alert("Error: " + data.error);
        } else {
            logToServer('info', 'Method saved successfully!')
            alert("Method saved successfully!");
        }
    })
    .catch(error => logToServer('critical', 'Fetch error: ${error}'));
}

function getCSRFToken() {
    let cookieValue = null;
    if (document.cookie) {
        const cookies = document.cookie.split(";");
        cookies.forEach(cookie => {
            const trimmed = cookie.trim();
            if (trimmed.startsWith("csrftoken=")) {
                cookieValue = trimmed.substring("csrftoken=".length);
            }
        });
    }
    return cookieValue;
}

function checkFolderExists(methodName) {
    return fetch(`/check_folder/?folder_name=${encodeURIComponent(methodName)}`)
        .then(response => response.json())
        .then(data => {
            if (data.exists) {
                alert("Folder already exists! Change method name.");
                return false;
            }
            return true;
        })
        .catch(error => {
            console.error("Error:", error);
            return false;
        });
}