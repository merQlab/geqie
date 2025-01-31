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
    const response = await fetch(updateListUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    });

    if (response.ok) {
        const data = await response.json();
        alert(data.message);
    } else {
        const error = await response.json();
        alert(`Error: ${error.error}`);
    }
    
    const name = document.getElementById('methodName');
    if (name === "") {
        alert("Method name is empty!");
        return;
    }
});