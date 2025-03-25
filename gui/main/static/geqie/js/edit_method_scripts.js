document.addEventListener("DOMContentLoaded", function () {
    logToServer('debug', 'DOM fully loaded. Setting up event listeners for methodSelect.');

    const methodSelect = document.getElementById('methodSelect');
    const testMethodSelect = document.getElementById('testMethodSelect');

    editorsList = [
        { field: "init", editorId: "initEditor", contentId: "initContent" },
        { field: "map", editorId: "mapEditor", contentId: "mapContent" },
        { field: "data", editorId: "dataEditor", contentId: "dataContent" },
        { field: "retrieve", editorId: "retrieveEditor", contentId: "retrieveContent" }
    ];

    editorsList.forEach(item => {
        item.textarea = document.getElementById(item.contentId);
        ace.require("ace/ext/language_tools");
        item.editor = ace.edit(item.editorId);
        item.editor.session.setMode("ace/mode/python");
        // item.editor.setTheme("ace/theme/merbivore_soft");
        item.editor.setTheme("ace/theme/xcode");
        item.editor.setFontSize("13px");
        item.editor.setShowPrintMargin(false);
        item.editor.setOptions({
            enableBasicAutocompletion: true,
            enableSnippets: true,
            enableLiveAutocompletion: true
        });
        item.editor.setValue(item.textarea.value, -1);
    });

    methodSelect.addEventListener("change", function () {
        const methodName = methodSelect.value;
        if (!methodName) {
            editorsList.forEach(item => {
                item.textarea.value = "";
                item.editor.setValue("", -1);
            });
            logToServer('debug', 'No method selected, clearing content fields.');
            return;
        }

        logToServer('debug', `Method selected: ${methodName}. Fetching method data...`);
        fetch(`/get-method/${methodName}/`)
            .then(response => {
                logToServer('debug', `Fetch initiated for method: ${methodName}`);
                return response.json();
            })
            .then(data => {
                editorsList.forEach(item => {
                    const value = data[item.field] || "No content found.";
                    item.textarea.value = value;
                    item.editor.setValue(value, -1);
                });
                logToServer('info', `Fetched method data for ${methodName}`);
            })
            .catch(error => logToServer('critical', `Error fetching method data: ${error}`));
    });

    testMethodSelect.addEventListener('change', () => {
        const selectedOption = testMethodSelect.options[testMethodSelect.selectedIndex];
        displayedName = selectedOption.text;
    });
});

let editorsData = [];
let editorsList = [];

document.addEventListener("DOMContentLoaded", function() {
    editorsData = [
        {
            contentId: "addInitContent",
            editorId: "addInitEditor",
            value: `import numpy as np
from qiskit.quantum_info import Statevector

def init(n_qubits: int) -> Statevector:    
    qubits_in_superposition = # place number of qubits in superposition
    base_state = np.zeros(2**qubits_in_superposition, dtype=int)    
    base_state[0] = 1    
    state = np.tile(base_state, 2**(n_qubits - qubits_in_superposition))
    return Statevector(state)`
        },
        {
            contentId: "addMapContent",
            editorId: "addMapEditor",
            value: `import numpy as np
from qiskit.quantum_info import Operator

def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:    
    p = image[u, v]  
    # Provide your own unitary matrix for map operator
    return Operator(map_operator)`
        },
        {
            contentId: "addDataContent",
            editorId: "addDataEditor",
            value: `import numpy as np
from qiskit.quantum_info import Statevector

def data(u: int, v: int, R: int, image: np.ndarray) -> Statevector:    
    m = u * image.shape[0] + v    
    data_vector = np.zeros(2**(2 * R))    
    data_vector[m] = 1    
    return Statevector(data_vector)`
        },
        {
            contentId: "addRetrieveContent",
            editorId: "addRetrieveEditor",
            value: `import numpy as np
import json
 
def retrieve(results: str) -> np.ndarray:
    """
    Decodes an image from quantum state measurement results.
    """
    state_length = len(next(iter(results)))
    # color_qubits = set qubits used for color encoding
    number_of_position_qubits = state_length - color_qubits
    x_qubits = number_of_position_qubits // 2
    y_qubits = number_of_position_qubits // 2
    image_shape = (2**x_qubits, 2**y_qubits)
    
    # Provide your own code here...
    return reconstructed_image`
        }
    ];

    editorsData.forEach(item => {
        const textarea = document.getElementById(item.contentId);
        textarea.value = item.value;
        
        ace.require("ace/ext/language_tools");
        const editor = ace.edit(item.editorId);
        editor.session.setMode("ace/mode/python");
        // editor.setTheme("ace/theme/merbivore_soft");
        editor.setTheme("ace/theme/xcode");
        editor.setValue(item.value, -1);
        editor.setFontSize("13px");
        editor.setShowPrintMargin(false);
        editor.setAnimatedScroll(true);
        editor.setOptions({
            enableBasicAutocompletion: true,
            enableSnippets: true,
            enableLiveAutocompletion: true
        });
    });

    logToServer('debug', 'Default method content loaded for addNewMethod.');
});

let formData = new FormData();
let displayedName;

document.getElementById("imagePath").addEventListener("click", function() {
    document.getElementById("fileInput").click();
    logToServer('debug', 'Image path clicked, triggering file input click.');
});

document.getElementById("fileInput").addEventListener("change", function() {
    const filePathInput = document.getElementById("filePath");
    if (this.files && this.files[0]) {
        filePathInput.value = this.files[0].name;
        logToServer('info', `File selected: ${this.files[0].name}`);
    } else {
        logToServer('warning', 'No file selected on fileInput change.');
    }
});

document.getElementById("addNewMethod").addEventListener("click", async function () {
    logToServer('info', 'User triggered addNewMethod.');
    const methodName = document.getElementById("methodName").value;
    const addNewMethodBtn = document.getElementById('addNewMethod');
    const loadingGif = document.getElementById('addLoadingGif');

    const canProceed = await checkFolderExists(methodName);
    if (!canProceed) {
        logToServer('info', `Folder check failed for method: ${methodName}`);
        return;
    }
    
    loadingGif.style.display = 'inline-block';
    addNewMethodBtn.disabled = true;
    
    updateTextareas(editorsData);

    await saveMethod(true, true, methodName, "methodName", "addInitContent", "addMapContent", "addDataContent", "addRetrieveContent", canProceed);

    const images = await fetchAllImageFiles();

    await startTest(methodName, images, 'simulate', '1024', true, false);

    addNewMethodBtn.disabled = false;
    loadingGif.style.display = 'none';
});

function updateTextareas(editorsList) {
    editorsList.forEach(item => {
      const editor = ace.edit(item.editorId);
      document.getElementById(item.contentId).value = editor.getValue();
    });
}

// document.getElementById("save").addEventListener("click", function () {
//     logToServer('info', 'Method saved.');
//     saveMethod(false, false, document.getElementById("methodSelect").value, "methodSelect", "initContent", "mapContent", "dataContent");
// });

document.getElementById("saveAsNew").addEventListener("click", async function () {
    let userInput = prompt("Method name:");
    const addNewMethodBtn = document.getElementById('saveAsNew');
    const loadingGif = document.getElementById('editLoadingGif');

    if (userInput === null) {
        alert("Canceled");
        logToServer('info', 'Save as new cancelled by user.');
    } else if (userInput.trim() === "") {
        alert("Name is empty");
        logToServer('warning', 'User attempted to save as new with an empty name.');
    } else {
        logToServer('info', 'User triggered saveAsNew.');

        loadingGif.style.display = 'inline-block';
        addNewMethodBtn.disabled = true;

        const methodName = userInput.trim()
        const canProceed = await checkFolderExists(methodName);

        if (!canProceed) {
            logToServer('info', `Folder check failed for method: ${methodName}`);
            addNewMethodBtn.disabled = false;
            loadingGif.style.display = 'none';
            return;
        }

        updateTextareas(editorsList);
        
        await saveMethod(false, true, methodName, "methodSelect", "initContent", "mapContent", "dataContent", "retrieveContent", canProceed);

        const images = await fetchAllImageFiles();

        await startTest(methodName, images, 'simulate', '1024', true, false);

        addNewMethodBtn.disabled = false;
        loadingGif.style.display = 'none';
    }
});

async function saveMethod(addNew, isNew, saveName, method, init, map, data, retrieve, isFolderExist) {
    let methodName = document.getElementById(method).value;
    let initContent = document.getElementById(init).value;
    let mapContent = document.getElementById(map).value;
    let dataContent = document.getElementById(data).value;
    let retrieveContent = document.getElementById(retrieve).value;

    logToServer('info', `saveMethod called with methodName="${methodName}", isNew=${isNew}, addNew=${addNew}, saveName="${saveName}"`);

    if(!addNew && isNew) {
        methodName = saveName;
    }

    if (!methodName) {
        alert("Please select a method first.");
        logToServer('warning', 'saveMethod aborted: no method selected.');
        return;
    }

    if (!isFolderExist) {
        return;
    }

    try {
        const response = await fetch("/save-method/", {
            method: "POST",
            credentials: "same-origin",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCSRFToken(),
            },
            body: JSON.stringify({
                method_name: methodName,
                init: initContent,
                map: mapContent,
                data: dataContent,
                retrieve: retrieveContent,
                is_new: isNew,
                save_name: saveName,
                add_new: addNew,
            }),
        });
        logToServer('info', `saveMethod response received for method: ${methodName}`);

        const result = await response.json();
        if (result.error) {
            logToServer('warning', `Error saving method: ${result.error}`);
            alert("Error: " + result.error);
        } else {
            logToServer('info', 'Method saved successfully.');
        }
    } catch (error) {
        logToServer('critical', `Fetch error in saveMethod: ${error}`);
    }
}

function getCSRFToken() {
    let cookieValue = null;
    if (document.cookie) {
        const cookies = document.cookie.split(";");
        cookies.forEach(cookie => {
            const trimmed = cookie.trim();
            if (trimmed.startsWith("csrftoken=")) {
                cookieValue = trimmed.substring("csrftoken=".length);
                logToServer('debug', `CSRF token retrieved: ${cookieValue}`);
            }
        });
    } else {
        logToServer('warning', 'No cookies found when attempting to retrieve CSRF token.');
    }
    return cookieValue;
}

function checkFolderExists(methodName) {
    logToServer('debug', `Checking if folder exists for method: ${methodName}`);
    return fetch(`/check_folder/?folder_name=${encodeURIComponent(methodName)}`)
        .then(response => response.json())
        .then(data => {
            if (data.exists) {
                alert("Folder already exists! Change method name.");
                logToServer('info', `Folder already exists for method: ${methodName}`);
                return false;
            }
            logToServer('info', `Folder does not exist for method: ${methodName}. Proceeding.`);
            return true;
        })
        .catch(error => {
            logToServer('error', `Error checking folder existence: ${error}`);
            return false;
        });
}

async function fetchAllImageFiles() {
    const response = await fetch('/get-all-images/');
    if (!response.ok) {
        logToServer('error', 'Error downloading images');
        return;
    }
    const data = await response.json();
    const imageFiles = [];

    data.images.forEach(img => {
        const binaryString = atob(img.data);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: img.type });
        const file = new File([blob], img.name, { type: img.type });
        imageFiles.push(file);
    });

    return imageFiles;
}

document.getElementById("startTest").addEventListener("click", async function () {
    const startTestBtn = document.getElementById('startTest');
    const loadingGif = document.getElementById('testLoadingGif');
    const shotsElement = document.getElementById('shots');
    const images = fileInput.files[0];

    logToServer('info', `Start test: method=${displayedName}, images=${fileInput.files.length}, shots=${shotsElement.value}`);

    if (!displayedName) {
        alert("Please select an method first.");
        return;
    }

    if (fileInput.files.length == 0) {
        alert("Please select an image first.");
        return;
    }

    startTestBtn.disabled = true;
    loadingGif.style.display = 'inline-block';

    try {
        const responseData = await startTest(displayedName, [images], 'simulate', shotsElement.value, false, true);
        
        if (responseData) {
            logToServer('info', `Response data=${responseData}`);
            const testResultInput = document.getElementById('testResult');
            testResultInput.innerHTML = "";

            for (const [fileName, result] of Object.entries(responseData.processed)) {
                const listItem = document.createElement("li");
                listItem.className = "list-group-item";

                const strongResultText = document.createElement("strong");
                strongResultText.textContent = "Result: ";
                const resultValueText = document.createTextNode(result);

                listItem.appendChild(strongResultText);
                listItem.appendChild(resultValueText);
                testResultInput.appendChild(listItem);

                if (responseData.retrieved_image) {
                    const retrievedImg = document.createElement("img");
                    retrievedImg.src = "data:image/png;base64," + responseData.retrieved_image[fileName];
                    const boxRight = document.querySelector(".box-right");
                    boxRight.appendChild(retrievedImg);
                }
            }
            
            const reader = new FileReader();
            reader.onload = function (e) {
                const img = document.createElement("img");
                img.src = e.target.result;
                const boxLeft = document.querySelector(".box-left");
                boxLeft.appendChild(img);
            };
            reader.readAsDataURL(images);
        } else {
            logToServer('warning', "No data received or request failed");
        }
    } catch (error) {
        logToServer('critical', `Response data=${error}`);
    }
    
    startTestBtn.disabled = false;
    loadingGif.style.display = 'none';
});

async function startTest(selected_method, images, computer, shots, is_test, is_retrieve) {
    logToServer('info', `Start test: method=${selected_method}, images=${images.length}, computer=${computer}, shots=${shots}, is_test=${is_test}, is_retrieve=${is_retrieve}`);
    
    const formData = new FormData();
    images.forEach(image => {
        formData.append('images[]', image);
    });
    formData.append('selected_method', selected_method);
    formData.append('computer', computer);
    formData.append('shots', shots);
    formData.append('is_test', is_test);
    formData.append('is_retrieve', is_retrieve);

    try {
        const response = await fetch(startTestUrl, {
            method: "POST",
            credentials: "same-origin",
            body: formData,
            headers: {
                "X-CSRFToken": "{{ csrf_token }}",
            },
        });

        if (response.ok) {
            const data = await response.json();
            logToServer('info', `Response ok: ${JSON.stringify(data, null, 2)}`);
            isResponseOk = true;

            if (Object.keys(data.processed).length === 0) {
                alert("Photo processing error with this method.");
                return null;
            }
            if (is_test === false && Object.keys(data.image).length === 0) {
                alert("Image is empty.");
            }
            if (is_test === false && Object.keys(data.retrieved_image).length === 0) {
                alert("Retrieved image is empty.");
            }

            alert("Executed successfully!");
            return data;
        } else {
            isResponseOk = false;
            const errorData = await response.json();
            logToServer('error', `Received errorData: ${JSON.stringify(errorData)}`);
            const errorMessage = errorData.error || "";
            logToServer('error', `Extracted errorMessage: ${JSON.stringify(errorMessage)}`);

            if (errorMessage.includes("Command failed")) {
                alert("Experiment failed due to a command execution error. Please check your method implementation.");
            } else {
                alert("Failed to start experiment: " + errorMessage);
            }
            return null;
        }
    } catch (error) {
        logToServer('critical', `Error starting experiment: ${error.message || error}`);
        return null;
    }
}