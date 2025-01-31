document.addEventListener('DOMContentLoaded', () => {
    const selectedSubMethod = document.getElementById('selected-submethod');
    const selectedComputer = document.getElementById('selected-computer');

    document.body.addEventListener('click', (event) => {
        const item = event.target.closest('.sub-item');
        if (!item) return;

        if (item.id.startsWith('method-')) {
            const methodName = item.textContent.trim();
            selectedSubMethod.textContent = methodName;
        }

        if (item.id.startsWith('subcomputer-')) {
            const subComputerName = item.textContent.trim();
            const computerName = item.getAttribute('computername');
            selectedComputer.textContent =  computerName + ' - ' + subComputerName || 'No computer selected';
        }
    });
});

let fileHandles = null;
let resultFolderPath = null;
let formData = new FormData();

document.getElementById('selectImage').addEventListener('click', () => {
    document.getElementById('imageFile').click();
});

document.getElementById('imageFile').addEventListener('change', async () => {
    formData = new FormData();
    const imageFile = document.getElementById('imageFile').files[0];

    if (!imageFile) return;

    formData.append('image', imageFile);
});

document.getElementById('resultFolder').addEventListener('click', async () => {
    try {
        const directoryHandle = await window.showDirectoryPicker();
        resultFolderPath = directoryHandle.name;
        alert(`Folder selected: ${resultFolderPath}`);
    } catch (error) {
        console.error("Folder selection cancelled:", error);
    }
});

document.getElementById('startExperiment').addEventListener('click', async () => {
    let isEmpty = true;
    for (let entry of formData.entries()) {
        isEmpty = false;
        break;
    }
    if (isEmpty) {
        alert("Please select photos first!");
        return;
    }

    if (!resultFolderPath) {
        alert("Please select result path!");
        return;
    }

    const selectedMethod = document.getElementById('selected-submethod').textContent.trim();
    if (selectedMethod === 'No method') {
        alert("Please select method!");
        return;
    }

    formData.append('result_folder', resultFolderPath);
    formData.append('selected_method', selectedMethod);
    const shots = document.getElementById('shots').value;
    formData.append('shots', shots);

    try {
        const response = await fetch(startExperimentUrl, {
            method: "POST",
            body: formData,
            headers: {
                "X-CSRFToken": "{{ csrf_token }}", 
            },
        });

        if (response.ok) {
            const data = await response.json();
            const resultsList = document.getElementById("results-list");

            //resultsList.innerHTML = "";

            if (data && Object.keys(data).length > 0) {
                for (const [fileName, result] of Object.entries(data)) {
                    const listItem = document.createElement("li");
                    listItem.className = "list-group-item";
                    listItem.textContent = `File: ${fileName}, Result: ${JSON.stringify(result)}`;
                    resultsList.appendChild(listItem);
                }
            } else {
                const noResultsItem = document.createElement("li");
                noResultsItem.className = "list-group-item text-muted";
                noResultsItem.textContent = "No results";
                resultsList.appendChild(noResultsItem);
            }
        } else {
            const errorData = await response.json();
            alert("Failed to start experiment: " + errorData.error);
        }
    } catch (error) {
        console.error("Error starting experiment:", error);
    }
});