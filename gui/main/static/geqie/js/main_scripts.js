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

    logToServer('debug', `Selected ${files.length} images: ${[...files].map(f => f.name).join(', ')}`);

    for (let i = 0; i < files.length; i++) {
        formData.append('images[]', files[i]);
    }
});

document.getElementById('addExperiment').addEventListener('click', () => {
    const files = document.getElementById('imageFile').files;
    const selectedMethod = document.getElementById('selected-submethod').textContent.trim();
    const selectedComputer = document.getElementById('selected-computer').textContent.trim();
    const shots = document.getElementById('shots').value;

    logToServer('debug', `Trying to add an experiment: method=${selectedMethod}, computer=${selectedComputer}, shots=${shots}, number of files=${files.length}`);

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
    logToServer('debug', `Experiment table updated, number of experiments: ${experiments.length}`);
}

function removeExperiment(index) {
    logToServer('debug', `Deleting an experiment by index ${index}`);
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
    const saveToCSVCheckbox = document.getElementById('saveToCSV');
    saveToCSVCheckbox.disabled = true;

    let allResults = [];
    let mainDirHandle = null;
    let methodDirHandles = {};

    if (saveToCSVCheckbox.checked) {
        try {
            mainDirHandle = await window.showDirectoryPicker();
            const uniqueMethods = [...new Set(experiments.map(exp => exp.method))];
            
            for (const method of uniqueMethods) {
                try {
                    methodDirHandles[method] = await mainDirHandle.getDirectoryHandle(method, { create: true });
                } catch (error) {
                    logToServer('critical', `Error creating subfolder for method ${method}: ${error}`);
                    alert(`Failed to create a subfolder for the method ${method}.`);
                }
            }
        } catch (error) {
            logToServer('critical', `Failed to select folder: ${error}`);
            alert("Failed to select folder.");
            startExperimentBtn.disabled = false;
            saveToCSVCheckbox.disabled = false;
            return;
        }
    }

    try {
        const totalImages = experiments.reduce((sum, experiment) => sum + experiment.images.length, 0);
        animateProgress(10, 90, totalImages / 2 * experiments.length * 3000);

        const resultsList = document.getElementById("results-list");
        resultsList.innerHTML = '';

        for (const [index, experiment] of experiments.entries()) {
            logToServer('info', `Starting experiment ${index + 1}: method=${experiment.method}, computer=${experiment.computer}, shots=${experiment.shots}`);
            const formData = new FormData();
            formData.append('selected_method', experiment.method);
            experiment.images.forEach(image => {
                formData.append('images[]', image);
            });
            formData.append('method', experiment.method);
            formData.append('computer', experiment.computer);
            formData.append('shots', experiment.shots);
            formData.append('is_retrieve', "true");

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
                    const resultsToIterate = data.processed ? data.processed : data;

                    logToServer('info', `Response ok: ${JSON.stringify(resultsToIterate, null, 2)}`);

                    if (resultsToIterate && Object.keys(resultsToIterate).length > 0) {
                        for (const [fileName, result] of Object.entries(resultsToIterate)) {
                            const listItem = document.createElement("li");
                            listItem.className = "list-group-item";

                            const strongFileText = document.createElement("strong");
                            strongFileText.textContent = "File: ";
                            listItem.appendChild(strongFileText);
                            listItem.appendChild(document.createTextNode(fileName));
                            listItem.appendChild(document.createElement("br"));

                            const strongResultText = document.createElement("strong");
                            strongResultText.textContent = "Result: ";
                            listItem.appendChild(strongResultText);
                            listItem.appendChild(document.createTextNode(result));
                            listItem.appendChild(document.createElement("br"));

                            const originalImageBase64 = data.image?.[fileName] || null;
                            const retrievedImageBase64 = data.retrieved_image?.[fileName] || null;

                            if (originalImageBase64) {
                                const originalImg = document.createElement("img");
                                originalImg.src = "data:image/png;base64," + originalImageBase64;
                                originalImg.alt = "Original image";
                                originalImg.style.width = "100%";
                                originalImg.style.imageRendering = "pixelated";
                                listItem.appendChild(originalImg);
                                listItem.appendChild(document.createElement("br"));
                            }

                            if (originalImageBase64 && retrievedImageBase64) {
                                const arrowDown = document.createElement("div");
                                arrowDown.innerHTML = "&#8595;";
                                arrowDown.style.fontSize = "24px";
                                arrowDown.style.textAlign = "center";
                                arrowDown.style.margin = "10px 0";
                                listItem.appendChild(arrowDown);
                            }
                            
                            if (retrievedImageBase64) {
                                const retrievedImg = document.createElement("img");
                                retrievedImg.src = "data:image/png;base64," + retrievedImageBase64;
                                retrievedImg.alt = "Retrieved image";
                                retrievedImg.style.width = "100%";
                                retrievedImg.style.imageRendering = "pixelated";
                                listItem.appendChild(retrievedImg);
                                listItem.appendChild(document.createElement("br"));
                            }

                            resultsList.appendChild(listItem);

                            allResults.push({
                                file: fileName,
                                result: result,
                                image: originalImageBase64,
                                retrieved_image: retrievedImageBase64,
                                method: experiment.method
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
                    logToServer('error', `Received errorData: ${JSON.stringify(errorData)}`);

                    const errorMessage = errorData.error || "";
                    logToServer('error', `Extracted errorMessage: ${JSON.stringify(errorMessage)}`);

                    if (errorMessage.includes("Command failed")) {
                        alert("Experiment " + (index + 1) + " failed due to a command execution error. Please check your method implementation.");
                    } else {
                        alert("Failed to start experiment " + (index + 1) + ": " + errorMessage);
                    }
                }
            } catch (error) {
                logToServer('critical', `Error starting experiment ${index + 1}: ${error.message || error}`);
            }
        }

        if (saveToCSVCheckbox.checked && allResults.length > 0 && mainDirHandle) {
            await saveResultsToFolder(allResults, mainDirHandle, methodDirHandles);
        }

        if (isResponseOk) {
            cancelAnimationFrame(animationFrameId);
            updateProgress(100);
        } else {
            cancelAnimationFrame(animationFrameId);
            updateProgress(0);
        }

    } finally {
        logToServer('info', `The experimentation process is complete. Results: ${JSON.stringify(allResults)}`);
        startExperimentBtn.disabled = false;
        saveToCSVCheckbox.disabled = false;
    }
});

async function getUniqueFileHandle(dirHandle, baseName, extension) {
    let fileName = baseName + extension;
    let counter = 0;
    while (true) {
        try {
            await dirHandle.getFileHandle(fileName, { create: false });
            counter++;
            fileName = `${baseName} (${counter})${extension}`;
        } catch (error) {
            logToServer('critical', `Error get unique file handle ${fileName}: ${error.message || error}`);
            return await dirHandle.getFileHandle(fileName, { create: true });
        }
    }
}

async function getUpscaledImageBlob(base64Image, scale) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width * scale;
            canvas.height = img.height * scale;
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(blob => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error("Failed to generate Blob from canvas."));
                }
            }, 'image/png');
        };
        img.onerror = reject;
        img.src = 'data:image/png;base64,' + base64Image;
    });
}

async function saveResultsToFolder(results, mainDirHandle, methodDirHandles) {
    for (const result of results) {
        const baseName = result.file.replace(/\.[^/.]+$/, "");
        try {
            const methodHandle = methodDirHandles[result.method];
            if (!methodHandle) {
                throw new Error(`No handle for the method ${result.method}`);
            }

            const workbook = new ExcelJS.Workbook();
            const worksheet = workbook.addWorksheet("Result");

            worksheet.getCell("A1").value = "File name:";
            worksheet.getCell("B1").value = result.file;
            worksheet.getCell("A2").value = "Result:";
            worksheet.getCell("B2").value = result.result;

            const scaleFactor = 64;

            const originalBlob = await getUpscaledImageBlob(result.image, scaleFactor);
            const originalBuffer = new Uint8Array(await originalBlob.arrayBuffer());

            const retrievedBlob = await getUpscaledImageBlob(result.retrieved_image, scaleFactor);
            const retrievedBuffer = new Uint8Array(await retrievedBlob.arrayBuffer());

            const originalImageId = workbook.addImage({
                buffer: originalBuffer,
                extension: 'png'
            });

            const retrievedImageId = workbook.addImage({
                buffer: retrievedBuffer,
                extension: 'png'
            });

            worksheet.addImage(originalImageId, {
                tl: { col: 0, row: 4 },
                ext: { width: 4 * scaleFactor, height: 4 * scaleFactor }
            });

            worksheet.getCell("E11").value = "→";
            worksheet.getCell("E11").alignment = { horizontal: "center", vertical: "middle" };

            worksheet.addImage(retrievedImageId, {
                tl: { col: 5, row: 4 },
                ext: { width: 4 * scaleFactor, height: 4 * scaleFactor }
            });

            const xlsxData = await workbook.xlsx.writeBuffer();
            const fileHandle = await getUniqueFileHandle(methodHandle, baseName, ".xlsx");
            const writable = await fileHandle.createWritable();
            const blob = new Blob([xlsxData], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
            await writable.write(blob);
            await writable.close();

        } catch (error) {
            logToServer('critical', `Error saving the file ${baseName}.xlsx: ${error.message || error}`);
            alert("An error occurred while saving XLSX files: " + error.message);
        }
    }
    alert("XLSX files have been successfully saved!");
}

function updateProgress(value) {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = value + '%';
    progressBar.setAttribute('aria-valuenow', value);
    progressBar.innerText = value + '%';
}

function animateProgress(from, to, duration) {
    logToServer('info', `Progress animation started: ${from}% -> ${to}% przez ${duration} ms`);
    const startTime = performance.now();
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        let progress = from + (to - from) * (elapsed / duration);
        if (progress > to) progress = to;
        updateProgress(Math.floor(progress));
        if (elapsed < duration) {
            animationFrameId = requestAnimationFrame(animate);
        } else {
            logToServer('info', `Progress animation finished on ${to}%`);
        }
    }
    animationFrameId = requestAnimationFrame(animate);
}