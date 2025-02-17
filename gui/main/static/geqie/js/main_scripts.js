let formData = new FormData();
let experiments = [];
let animationFrameId;
let isResponseOk;

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
    let formData = new FormData();
    const files = document.getElementById('imageFile').files;

    if (!files.length) return;

    for (let i = 0; i < files.length; i++) {
        formData.append('images[]', files[i]);
    }
});

document.getElementById('addExperiment').addEventListener('click', () => {
    const files = document.getElementById('imageFile').files;
    const selectedMethod = document.getElementById('selected-submethod').textContent.trim();
    const selectedComputer = document.getElementById('selected-computer').textContent.trim();
    const shots = document.getElementById('shots').value;

    if (files.length === 0) {
        alert("Please select an image first!");
        return;
    }

    if (selectedMethod === 'No method' || selectedMethod === '') {
        alert("Please select a method!");
        return;
    }

    if (shots <= 0) {
        alert("Please enter a valid number of shots greater than 0!");
        return;
    }

    const experiment = {
        images: Array.from(files),
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
        let imageNames = experiment.imageName;
        if (!imageNames && experiment.images && experiment.images.length > 0) {
            imageNames = experiment.images.map(image => image.name).join(', ');
        }

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>
                <div style="max-width: 320px; overflow-x: auto; white-space: nowrap;">
                    ${imageNames}
                </div>
            </td>
            <td style="vertical-align: middle;">${experiment.method}</td>
            <td style="vertical-align: middle;">${experiment.computer}</td>
            <td style="vertical-align: middle;">${experiment.shots}</td>
            <td style="vertical-align: middle;">
                <img 
                    src="${trashIconPath}" 
                    alt="Trash Icon" 
                    style="width:20px;height:20px;cursor:pointer;" 
                    onclick="removeExperiment(${index})"
                >
            </td>
        `;
        tableBody.appendChild(row);
    });
}

function removeExperiment(index) {
    experiments.splice(index, 1);
    updateExperimentsTable();
}


document.getElementById('startExperiment').addEventListener('click', async function(event) {
    event.preventDefault();

    if (experiments.length === 0) {
        alert("Please add experiments first!");
        return;
    }

    const startExperimentBtn = document.getElementById('startExperiment');
    startExperimentBtn.disabled = true;

    let allResults = [];

    try {
        const totalImages = experiments.reduce((sum, experiment) => sum + experiment.images.length, 0);
        animateProgress(10, 90, totalImages / 2 * experiments.length * 3000);

        const resultsList = document.getElementById("results-list");
        resultsList.innerHTML = '';

        for (const [index, experiment] of experiments.entries()) {
            const formData = new FormData();
            formData.append('selected_method', experiment.method);
            experiment.images.forEach(image => {
                formData.append('images[]', image);
            });
            formData.append('method', experiment.method);
            formData.append('computer', experiment.computer);
            formData.append('shots', experiment.shots);

            try {
                const response = await fetch(startExperimentUrl, {
                    method: "POST",
                    credentials: "same-origin",
                    body: formData,
                    headers: {
                        "X-CSRFToken": "{{ csrf_token }}",
                    },
                });

                if (response.ok) {
                    const headerItem = document.createElement("li");
                    headerItem.className = "list-group-item fw-bold";
                    headerItem.textContent = `Experiment ${index + 1}:`;
                    resultsList.appendChild(headerItem);
                    const data = await response.json();
                    if (data && Object.keys(data).length > 0) {
                        for (const [fileName, result] of Object.entries(data)) {
                            const listItem = document.createElement("li");
                            listItem.className = "list-group-item";

                            const strongFileText = document.createElement("strong");
                            strongFileText.textContent = "File: ";
                            const fileNameText = document.createTextNode(fileName);

                            const strongResultText = document.createElement("strong");
                            strongResultText.textContent = "Result: ";
                            const resultValueText = document.createTextNode(JSON.stringify(result, null, 2));

                            listItem.appendChild(strongFileText);
                            listItem.appendChild(fileNameText);
                            listItem.appendChild(document.createElement("br"));
                            listItem.appendChild(strongResultText);
                            listItem.appendChild(resultValueText);
                            resultsList.appendChild(listItem);

                            allResults.push({
                                file: fileName,
                                result: JSON.stringify(result)
                            });
                        }
                    } else {
                        const noResultsItem = document.createElement("li");
                        noResultsItem.className = "list-group-item text-muted";
                        noResultsItem.textContent = "No results";
                        resultsList.appendChild(noResultsItem);
                    }
                    isResponseOk = true;
                } else {
                    isResponseOk = false;
                    const errorData = await response.json();
                    logToServer('error', `Received errorData: ${errorData}`);

                    const errorMessage = errorData.error || "";
                    logToServer('error', `Extracted errorMessage: ${errorMessage}`);

                    if (errorMessage.includes("Command failed")) {
                        alert("Experiment " + (index + 1) + " failed due to a command execution error. Please check your method implementation.");
                    } else {
                        alert("Failed to start experiment " + (index + 1) + ": " + errorMessage);
                    }
                }
            } catch (error) {
                logToServer('error', `Error starting experiment ${index + 1}: ${error}`);
            }
        }

        if (isResponseOk) {
            cancelAnimationFrame(animationFrameId);
            updateProgress(100);
        } else {
            cancelAnimationFrame(animationFrameId);
            updateProgress(0);
        }

        const saveToCSV = document.getElementById("saveToCSV").checked;
        if (saveToCSV && allResults.length > 0) {
          const csvData = convertToCSV(allResults);
          downloadCSV(csvData, 'results.csv');
        }
        
    } finally {
        startExperimentBtn.disabled = false;
    }
});

function convertToCSV(data) {
    if (data.length === 0) return '';
    const header = Object.keys(data[0]).join(',') + '\n';
    const rows = data.map(row => Object.values(row).join(',')).join('\n');
    return header + rows;
}

function downloadCSV(csvData, filename = 'results.csv') {
    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

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
            animationFrameId = requestAnimationFrame(animate);
        }
    }
    animationFrameId = requestAnimationFrame(animate);
}