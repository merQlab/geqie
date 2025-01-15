document.addEventListener('DOMContentLoaded', () => {
    const selectedSubMethod = document.getElementById('selected-submethod');
    const selectedComputer = document.getElementById('selected-computer');

    document.body.addEventListener('click', (event) => {
        const item = event.target.closest('.sub-item');
        if (!item) return;

        if (item.id.startsWith('submethod-')) {
            const subMethodName = item.textContent.trim();
            const methodName = item.getAttribute('methodname');
            selectedSubMethod.textContent = methodName + ' - ' + subMethodName || 'No method selected';
        }

        if (item.id.startsWith('subcomputer-')) {
            const subComputerName = item.textContent.trim();
            const computerName = item.getAttribute('computername');
            selectedComputer.textContent =  computerName + ' - ' + subComputerName || 'No computer selected';
        }
    });
});

let fileHandles = null;
let selectedFolderPath = null;
let resultFolderPath = null;
let selectedMethod = null;

document.getElementById('selectFolder').addEventListener('click', async () => {
    try {
        fileHandles = await window.showOpenFilePicker({
            multiple: true,
            types: [{
                description: 'Images',
                accept: {'image/*': ['.jpg', '.jpeg', '.png', '.gif']}
            }]
        });
    } catch (error) {
        console.error("Folder selection cancelled:", error);
    }
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
    if (fileHandles === null) {
        alert("Please select a folder with photos first!");
        return;
    }
    if (!resultFolderPath) {
        alert("Please select result path!");
        return;
    }

    var selectedMethod = document.getElementById("selected-submethod");
    if (selectedMethod.textContent === "No method") {
        alert("Please select method!");
        return;
    }

    var selectedComputer = document.getElementById("selected-computer");
    if (selectedComputer.textContent === "No computer") {
        alert("Please select computer!");
        return;
    }

    try {
        const response = await fetch(startExperimentUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": "{{ csrf_token }}"
            },
            body: JSON.stringify({ folder: selectedFolderPath }),
        });

        if (response.ok) {
            alert("Folder path sent successfully!");
        } else {
            alert("Failed to start experiment");
        }
    } catch (error) {
        console.error("Error start experiment:", error);
    }
});