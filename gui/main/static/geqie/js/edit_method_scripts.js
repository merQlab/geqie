const methodSelect = document.getElementById('methodSelect');
const submethodSelect = document.getElementById('submethodSelect');
const initContent = document.getElementById('initContent');
const mapContent = document.getElementById('mapContent');
const dataContent = document.getElementById('dataContent');

methodSelect.addEventListener('change', function () {
    const selectedMethodId = this.value;

    Array.from(submethodSelect.options).forEach(option => {
        if (!option.value) {
            option.style.display = '';
        } else if (option.dataset.methodId === selectedMethodId) {
            option.style.display = '';
        } else {
            option.style.display = 'none';
        }
    });

    submethodSelect.value = '';
    resetDetails();
});

submethodSelect.addEventListener('change', function ()  {
    const selectedOption = this.options[this.selectedIndex];
    if (selectedOption) {
        const init = selectedOption.getAttribute('data-init');
        const map = selectedOption.getAttribute('data-map');
        const data = selectedOption.getAttribute('data-data');

        initContent.value = init || '';
        mapContent.value = map || '';
        dataContent.value = data || '';
    }
});

function resetDetails() {
    initContent.value = '';
    mapContent.value = '';
    dataContent.value = '';
}

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