let formData = new FormData();
let experiments = [];
let resultFolderPath = '';

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
            selectedComputer.textContent = (computerName && subComputerName) 
                ? computerName + ' - ' + subComputerName 
                : 'No computer selected';
        }
    });
});

document.getElementById('selectImage').addEventListener('click', () => {
    document.getElementById('imageFile').click();
});

document.getElementById('imageFile').addEventListener('change', async () => {
    formData = new FormData();
    const imageFile = document.getElementById('imageFile').files[0];

    if (!imageFile) return;
});

document.getElementById('addExperiment').addEventListener('click', () => {
    const imageFile = document.getElementById('imageFile').files[0];
    const selectedMethod = document.getElementById('selected-submethod').textContent.trim();
    const selectedComputer = document.getElementById('selected-computer').textContent.trim();
    const shots = document.getElementById('shots').value;

    if (!imageFile) {
        alert("Please select an image first!");
        return;
    }

    if (selectedMethod === 'No method' || selectedMethod === '') {
        alert("Please select a method!");
        return;
    }

    const experiment = {
        image: imageFile,
        imageName: imageFile.name,
        method: selectedMethod,
        computer: selectedComputer,
        shots: shots
    };

    experiments.push(experiment);
    updateExperimentsTable();
});

function updateExperimentsTable() {
    const tableBody = document.getElementById('experimentsTableBody');
    tableBody.innerHTML = '';

    experiments.forEach((experiment, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${experiment.imageName}</td>
            <td>${experiment.method}</td>
            <td>${experiment.computer}</td>
            <td>${experiment.shots}</td>
            <td><button class="btn btn-danger btn-sm" onclick="removeExperiment(${index})">Remove</button></td>
        `;
        tableBody.appendChild(row);
    });
}

function removeExperiment(index) {
    experiments.splice(index, 1);
    updateExperimentsTable();
}

document.getElementById('startExperiment').addEventListener('click', async function(event)  {
    event.preventDefault();

    if (experiments.length === 0) {
        alert("Please add experiments first!");
        return;
    }

    // if (!resultFolderPath) {
    //     alert("Please select result folder!");
    //     return;
    // }

    for (const [index, experiment] of experiments.entries()) {
        const formData = new FormData();
        formData.append('result_folder', resultFolderPath);
        formData.append('selected_method', experiment.method);
        formData.append('image', experiment.image);
        formData.append('method', experiment.method);
        formData.append('computer', experiment.computer);
        formData.append('shots', experiment.shots);

        updateProgress(10);
        animateProgress(10, 90, 3000);

        try {
            const response = await fetch(startExperimentUrl, {
                method: "POST",
                body: formData,
                headers: {
                    "X-CSRFToken": "{{ csrf_token }}",
                },
            });
            updateProgress(90);

            const resultsList = document.getElementById("results-list");
            const headerItem = document.createElement("li");
            headerItem.className = "list-group-item fw-bold";
            headerItem.textContent = `Experiment ${index + 1}:`;
            resultsList.appendChild(headerItem);

            if (response.ok) {
                const data = await response.json();
                if (data && Object.keys(data).length > 0) {
                    for (const [fileName, result] of Object.entries(data)) {
                        const listItem = document.createElement("li");
                        listItem.className = "list-group-item";
                        listItem.textContent = `File: ${fileName}, Result: ${JSON.stringify(result, null, 2)}`;
                        resultsList.appendChild(listItem);
                    }
                } else {
                    const noResultsItem = document.createElement("li");
                    noResultsItem.className = "list-group-item text-muted";
                    noResultsItem.textContent = "No results";
                    resultsList.appendChild(noResultsItem);
                }
                updateProgress(100);
            } else {
                const errorData = await response.json();
                updateProgress(0);
                alert("Failed to start experiment " + (index + 1) + ": " + errorData.error);
            }
        } catch (error) {
            console.error("Error starting experiment " + (index + 1) + ":", error);
        }
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

function updateProgress(value) {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = value + '%';
    progressBar.setAttribute('aria-valuenow', value);
    progressBar.innerText = value + '%';
}

function animateProgress(from, to, duration) {
    const startTime = performance.now();
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        let progress = from + (to - from) * (elapsed / duration);
        if (progress > to) progress = to;
        updateProgress(Math.floor(progress));
        if (elapsed < duration) {
            requestAnimationFrame(animate);
        }
    }
    requestAnimationFrame(animate);
}