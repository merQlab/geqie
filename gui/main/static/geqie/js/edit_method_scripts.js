const defaultMethodsContent = JSON.parse(document.getElementById('default_methods_content').textContent);
let editorsData = [];
let editorsList = [];
let formData = new FormData();
let displayedName;
let isResponseOk; 

function jobStatusUrl(jobId) {
    return `/job-status/${jobId}/`;
}

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

document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tab => {
    tab.addEventListener('shown.bs.tab', function (event) {
        const targetId = event.target.getAttribute('data-bs-target').substring(1);
        const editorItem = editorsList.find(item => item.field === targetId);
        if (editorItem && editorItem.editor) {
            editorItem.editor.resize();
        }
    });
});

document.addEventListener("DOMContentLoaded", function() {
    editorsData = [
        {
            contentId: "addInitContent",
            editorId: "addInitEditor",
            value: defaultMethodsContent.init
        },
        {
            contentId: "addMapContent",
            editorId: "addMapEditor",
            value: defaultMethodsContent.map
        },
        {
            contentId: "addDataContent",
            editorId: "addDataEditor",
            value: defaultMethodsContent.data
        },
        {
            contentId: "addRetrieveContent",
            editorId: "addRetrieveEditor",
            value: defaultMethodsContent.retrieve
        }
    ];

    editorsData.forEach(item => {
        const textarea = document.getElementById(item.contentId);
        textarea.value = item.value;

        ace.require("ace/ext/language_tools");
        const editor = ace.edit(item.editorId);
        editor.session.setMode("ace/mode/python");
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

    await startTest(methodName, [], 'simulate', '1024', true, true);

    addNewMethodBtn.disabled = false;
    loadingGif.style.display = 'none';
});


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

        await startTest(methodName, [], 'simulate', '1024', true, true);

        addNewMethodBtn.disabled = false;
        loadingGif.style.display = 'none';
    }
});

function updateTextareas(editorsList) {
    editorsList.forEach(item => {
      const editor = ace.edit(item.editorId);
      document.getElementById(item.contentId).value = editor.getValue();
    });
}

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
        return [];
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

    logToServer('info', `Start test click: method=${displayedName || '(none)'}, images=${fileInput.files.length}, shots=${shotsElement.value}`);

    if (!displayedName) {
        alert("Please select an method first.");
        return;
    }

    if (fileInput.files.length == 0) {
        alert("Please select an image first.");
        return;
    }

    const testResultList = document.getElementById('testResult');
    if (testResultList) testResultList.innerHTML = "";
    const boxLeft = document.querySelector(".box-left");
    const boxRight = document.querySelector(".box-right");
    if (boxLeft) boxLeft.innerHTML = "";
    if (boxRight) boxRight.innerHTML = "";

    const reader = new FileReader();
    reader.onload = function (e) {
        const img = document.createElement("img");
        img.src = e.target.result;
        if (boxLeft) boxLeft.appendChild(img);
    };
    reader.readAsDataURL(images);

    startTestBtn.disabled = true;
    loadingGif.style.display = 'inline-block';

    try {
        const data = await startTest(displayedName, [images], 'simulate', shotsElement.value, true, true);
    } catch (error) {
        logToServer('critical', `Start test click error: ${error?.message || error}`);
    }

    startTestBtn.disabled = false;
    loadingGif.style.display = 'none';
});


async function startTest(selected_method, images, computer, shots, is_test, is_retrieve) {
    logToServer('info', `Start test (jobs): method=${selected_method}, images=${images.length}, computer=${computer}, shots=${shots}, is_test=${is_test}, is_retrieve=${is_retrieve}`);

    const formData = new FormData();
    images.forEach(image => {
        formData.append('images[]', image);
    });
    formData.append('selected_method', selected_method);
    formData.append('computer', computer);
    formData.append('shots', shots);
    formData.append('is_test', is_test);
    formData.append('is_retrieve', is_retrieve);

    const testResultList = document.getElementById('testResult');

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

            if (!data.jobs || !Array.isArray(data.jobs) || data.jobs.length === 0) {
                const noResultsItem = document.createElement("li");
                noResultsItem.className = "list-group-item text-muted";
                noResultsItem.textContent = "No jobs returned";
                if (testResultList) testResultList.appendChild(noResultsItem);
                return data;
            }

            const headerItem = document.createElement("li");
            headerItem.className = "list-group-item fw-bold";
            headerItem.textContent = `Test: ${selected_method}`;
            if (testResultList) testResultList.appendChild(headerItem);

            for (const job of data.jobs) {
                await pollJobAndRender(job, testResultList, selected_method);
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

async function pollJobAndRender(job, resultsList, methodName) {
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
                const em = document.createElement("div");
                em.className = "text-danger mt-1";
                em.textContent = s.error;
                listItem.appendChild(document.createElement("br"));
                listItem.appendChild(em);
            }
            break;
        }

        if (s.status === "done") {
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

                            resultText = JSON.stringify(obj);
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

            if (s.original_url) {
                const originalImg = document.createElement("img");
                originalImg.src = s.original_url;
                originalImg.alt = "Original image";
                originalImg.style.width = "100%";
                originalImg.style.imageRendering = "pixelated";
                listItem.appendChild(document.createElement("br"));
                listItem.appendChild(originalImg);
            }

            if (s.original_url && s.retrieved_url) {
                const arrowDown = document.createElement("div");
                arrowDown.innerHTML = "&#8595;";
                arrowDown.style.fontSize = "24px";
                arrowDown.style.textAlign = "center";
                arrowDown.style.margin = "10px 0";
                listItem.appendChild(arrowDown);
            }

            if (s.retrieved_url) {
                const retrievedImg = document.createElement("img");
                retrievedImg.src = s.retrieved_url;
                retrievedImg.alt = "Retrieved image";
                retrievedImg.style.width = "100%";
                retrievedImg.style.imageRendering = "pixelated";
                listItem.appendChild(retrievedImg);

                const boxRight = document.querySelector(".box-right");
                if (boxRight) {
                    const thumb = document.createElement("img");
                    thumb.src = s.retrieved_url;
                    boxRight.appendChild(thumb);
                }
            }
            break;
        }
    }
}