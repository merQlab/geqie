window.logToServer = function(level, message) {
    fetch('/api/log/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            level: level,
            message: message,
        }),
    })
    .then(response => {
        if (!response.ok) {
            console.error("Failed to send the log to the server.");
        }
    })
    .catch(error => {
        console.error("Error sending log to server: ", error);
    });
}