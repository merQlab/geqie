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
    const totalImages = experiments.reduce((sum, experiment) => sum + experiment.images.length, 0);
    let finishedImages = 0;

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
        initProgress();

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

                    if (!data.jobs || !Array.isArray(data.jobs) || data.jobs.length === 0) {
                        const noResultsItem = document.createElement("li");
                        noResultsItem.className = "list-group-item text-muted";
                        noResultsItem.textContent = "No jobs returned";
                        resultsList.appendChild(noResultsItem);
                    } else {
                        for (const job of data.jobs) {
                            await pollJobAndRender(job, resultsList, experiment.method, allResults, saveToCSVCheckbox.checked);
                            // Count each finished job (image) rather than counting experiments.
                            finishedImages += 1;
                            const progress = Math.min(100, +(finishedImages / totalImages * 100).toFixed(1));
                            updateProgress(progress);
                            if (progress == 100) {
                                stopProgress();
                            }
                        }
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

    } finally {
        logToServer('info', `The experimentation process is complete. Results: ${JSON.stringify(allResults)}`);
        startExperimentBtn.disabled = false;
        saveToCSVCheckbox.disabled = false;
    }
});

function jobStatusUrl(jobId) {
    return `/job-status/${jobId}/`;
}

function stringifyAndSort(obj) {
    const entries = Object.entries(obj)
        .sort(([a], [b]) => a.localeCompare(b));

    const lines = entries.map(
        ([k, v]) => `"${k}": ${JSON.stringify(v)}`
    );

    return `{${lines.join(", ")}}`;
}

async function pollJobAndRender(job, resultsList, methodName, allResults, needBase64ForSave) {
    const listItem = document.createElement("li");
    listItem.className = "list-group-item";

    const strongFileText = document.createElement("strong");
    strongFileText.textContent = "File: ";
    listItem.appendChild(strongFileText);
    listItem.appendChild(document.createTextNode(job.file));
    listItem.appendChild(document.createElement("br"));

    const statusStrong = document.createElement("strong");
    statusStrong.textContent = "Status: ";
    listItem.appendChild(statusStrong);
    const statusText = document.createElement("span");
    statusText.textContent = "queued";
    listItem.appendChild(statusText);

    resultsList.appendChild(listItem);

    let attempts = 0;
    while (true) {
        await new Promise(r => setTimeout(r, Math.min(2500, 1000 + attempts * 300)));
        attempts++;

        let resp;
        try {
            resp = await fetch(jobStatusUrl(job.job_id), { credentials: "same-origin" });
        } catch (e) {
            statusText.textContent = `network error: ${e.message || e}`;
            continue;
        }

        if (!resp.ok) {
            statusText.textContent = `http ${resp.status}`;
            continue;
        }

        const s = await resp.json();
        statusText.textContent = s.status || "unknown";

        if (s.status === "error" || s.status === "failed") {
            if (s.error) {
                const details = document.createElement("details");
                const summary = document.createElement("summary");
                const error_message = document.createElement("div");

                error_message.className = "text-danger mt-1";
                error_message.style.paddingLeft = "4px";
                error_message.style.textAlign = "left";
                error_message.style.fontFamily = "monospace";
                error_message.style.whiteSpace = "pre-wrap";
                error_message.textContent = s.error;
                
                summary.textContent = "Error details";
                summary.style.cursor = "pointer";
                
                details.appendChild(summary);
                details.appendChild(error_message);
                listItem.appendChild(document.createElement("br"));
                listItem.appendChild(details);
                stopProgress();
            }
            break;
        }

        if (s.status === "done") {
            let resultObj = null;
            let resultText = "";
            if (s.output_json_url) {
                try {
                    const r = await fetch(s.output_json_url);
                    if (r.ok) {
                        const ct = r.headers.get("Content-Type") || "";
                        if (ct.includes("application/json")) {
                            let obj = await r.json();

                            if (Array.isArray(obj) &&
                                obj.length === 1 &&
                                Array.isArray(obj[0]) &&
                                obj[0].length === 2 &&
                                obj[0][0] === "counts" &&
                                typeof obj[0][1] === "object") {
                                obj = obj[0][1];
                            }
                            else if (obj && typeof obj === "object" && obj.counts && typeof obj.counts === "object") {
                                obj = obj.counts;
                            }

                            resultObj = obj;
                            resultText = stringifyAndSort(obj);
                        } else {
                            resultText = await r.text();
                        }

                    } else {
                        resultText = `Result available: ${s.output_json_url}`;
                    }
                } catch (e) {
                    resultText = "";
                }
            }

            if (resultText) {
                const strongResultText = document.createElement("strong");
                strongResultText.textContent = "Result: ";
                listItem.appendChild(document.createElement("br"));
                listItem.appendChild(strongResultText);

                const pre = document.createElement("pre");
                pre.style.whiteSpace = "pre";
                pre.style.overflowX = "auto";
                pre.style.maxWidth = "100%";
                pre.textContent = resultText;
                listItem.appendChild(pre);
            }

            if (s.original_url || s.retrieved_url) {
                const row = document.createElement("div");
                row.style.display = "flex";
                row.style.flexDirection = "row";
                row.style.alignItems = "center";
                row.style.gap = "10px"; // spacing
                row.style.margin = "10px 0";

                if (s.original_url) {
                    const originalImg = document.createElement("img");
                    originalImg.src = s.original_url;
                    originalImg.alt = "Original image";
                    originalImg.style.width = "50%";
                    originalImg.style.imageRendering = "pixelated";
                    row.appendChild(originalImg);
                }

                if (s.original_url && s.retrieved_url) {
                    const arrow = document.createElement("div");
                    arrow.innerHTML = "&#8594;";
                    arrow.style.fontSize = "24px";
                    arrow.style.textAlign = "center";
                    row.appendChild(arrow);
                }

                if (s.retrieved_url) {
                    const retrievedImg = document.createElement("img");
                    retrievedImg.src = s.retrieved_url;
                    retrievedImg.alt = "Retrieved image";
                    retrievedImg.style.width = "50%";
                    retrievedImg.style.imageRendering = "pixelated";
                    retrievedImg.style.filter = "invert(1)";
                    row.appendChild(retrievedImg);
                }

                listItem.appendChild(row);
            }
            let originalB64 = null;
            let retrievedB64 = null;
            if (needBase64ForSave) {
                try {
                    if (s.original_url) originalB64 = await fetchToBase64(s.original_url);
                    if (s.retrieved_url) retrievedB64 = await fetchToBase64(s.retrieved_url);
                } catch (e) {
                    logToServer('error', `Failed to fetch images as base64: ${e.message || e}`);
                }
            }

            allResults.push({
                file: job.file,
                result: resultText || (resultObj ? JSON.stringify(resultObj) : ""),
                image: originalB64,
                retrieved_image: retrievedB64,
                method: methodName
            });
            break;
        }
    }
}

async function fetchToBase64(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const blob = await r.blob();
    return await blobToBase64(blob);
}

function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = () => reject(new Error("Failed to read blob"));
        reader.onload = () => {
            const dataUrl = reader.result;
            const base64 = String(dataUrl).split(",")[1] || "";
            resolve(base64);
        };
        reader.readAsDataURL(blob);
    });
}

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
            worksheet.getCell("B2").value = typeof result.result === 'string' ? result.result : JSON.stringify(result.result);

            const scaleFactor = 64;

            if (result.image) {
                const originalBlob = await getUpscaledImageBlob(result.image, scaleFactor);
                const originalBuffer = new Uint8Array(await originalBlob.arrayBuffer());
                const originalImageId = workbook.addImage({ buffer: originalBuffer, extension: 'png' });
                worksheet.addImage(originalImageId, {
                    tl: { col: 0, row: 4 },
                    ext: { width: 4 * scaleFactor, height: 4 * scaleFactor }
                });
            }

            if (result.retrieved_image) {
                worksheet.getCell("E11").value = "→";
                worksheet.getCell("E11").alignment = { horizontal: "center", vertical: "middle" };

                const retrievedBlob = await getUpscaledImageBlob(result.retrieved_image, scaleFactor);
                const retrievedBuffer = new Uint8Array(await retrievedBlob.arrayBuffer());
                const retrievedImageId = workbook.addImage({ buffer: retrievedBuffer, extension: 'png' });
                worksheet.addImage(retrievedImageId, {
                    tl: { col: 5, row: 4 },
                    ext: { width: 4 * scaleFactor, height: 4 * scaleFactor }
                });
            }

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
    progressBar.innerText = value + '%';
    progressBar.style.width = value + '%';
    progressBar.setAttribute('aria-valuenow', value);
}

function initProgress() {
    const progressBar = document.getElementById('progress-bar');
    progressBar.classList.add('progress-bar-animated', 'progress-bar-striped');
    progressBar.innerText = 'Initializing...';
    progressBar.style.width = '100%';
}

function stopProgress() {
    const progressBar = document.getElementById('progress-bar');
    progressBar.classList.remove('progress-bar-animated', 'progress-bar-striped');
    progressBar.style.width = '100%';
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