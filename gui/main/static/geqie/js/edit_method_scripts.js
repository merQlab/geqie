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
            .catch(error => console.error("Error fetching method data:", error));
    });
});

document.getElementById('testMethod').addEventListener('click', async () => {
    try {
        const fileHandles = await window.showOpenFilePicker({
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

document.getElementById('addNewMethod').addEventListener('click', async () => {
    
});

document.getElementById("save").addEventListener("click", function () {
    saveMethod(false);
});

document.getElementById("saveAsNew").addEventListener("click", function () {
    let userInput = prompt("Method name:");
    if (userInput === null) {
        alert("Canceled");
    } else if (userInput.trim() === "") {
        alert("Name is empty");
    } else {
        saveMethod(true, userInput.trim());
    }
});

function saveMethod(isNew, saveName) {
    let methodName = document.getElementById("methodSelect").value;
    let initContent = document.getElementById("initContent").value;
    let mapContent = document.getElementById("mapContent").value;
    let dataContent = document.getElementById("dataContent").value;

    if (!methodName) {
        alert("Please select a method first.");
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
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
        } else {
            alert("Method saved successfully!");
        }
    })
    .catch(error => console.error("Fetch error:", error));
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