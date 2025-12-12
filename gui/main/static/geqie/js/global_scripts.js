window.logToServer = function(level, message) {
    const payload = { level, message };
    fetch('/logs/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    })
    .then(response => {
        if (!response.ok) {
            console.error(`Failed to send log to server. Status: ${response.status}`);
        } else {
            console.debug(`Log sent to server: [${level}] ${message}`);
        }
    })
    .catch(error => {
        console.error("Error sending log to server: ", error);
    });
};