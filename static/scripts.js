document.addEventListener('DOMContentLoaded', function() {
    const chatMessagesDiv = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const attachToggle = document.getElementById('attach-toggle');
    const uploadMenu = document.getElementById('upload-menu');
    const takePhotoButton = document.getElementById('take-photo-button');
    const uploadFileInput = document.getElementById('upload-file-input');

    // Function to append messages to the chat interface
    function appendMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.innerHTML = message; // Use innerHTML to allow for formatted text (like the XAI report)
        
        messageDiv.appendChild(contentDiv);
        chatMessagesDiv.appendChild(messageDiv);
        chatMessagesDiv.scrollTop = chatMessagesDiv.scrollHeight; // Scroll to bottom
    }

    // Function to handle sending messages/queries
    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        appendMessage('user', query);
        userInput.value = ''; // Clear input

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });
            const data = await response.json();

            if (data.success) {
                appendMessage('ai', data.response);
            } else {
                appendMessage('ai', `Error: ${data.error || 'Something went wrong.'}`);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            appendMessage('ai', 'Oops! Could not connect to the server. Please try again.');
        }
    }

    // Event listener for the Send button
    sendButton.addEventListener('click', sendMessage);

    // Event listener for Enter key in the textarea
    userInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) { // Shift+Enter for new line
            event.preventDefault(); // Prevent default new line
            sendMessage();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto'; // Reset height
        this.style.height = (this.scrollHeight) + 'px'; // Set to scroll height
    });

    // --- Attachment Menu Logic ---

    // 1. Check for Camera Access
    function checkCameraSupport() {
        // navigator.mediaDevices is the standard way to check for camera/mic access
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            // Check if 'video' source is available
            navigator.mediaDevices.enumerateDevices()
                .then(devices => {
                    const hasCamera = devices.some(device => device.kind === 'videoinput');
                    if (hasCamera) {
                        takePhotoButton.style.display = 'flex'; // Show the button if camera found
                    }
                })
                .catch(err => {
                    console.error("Error accessing media devices:", err);
                    // If error occurs, assume no reliable camera access, keep button hidden
                });
        }
    }
    
    // Run camera check when the page loads
    checkCameraSupport();


    // 2. Menu Toggle Logic
    attachToggle.addEventListener('click', function() {
        const isExpanded = attachToggle.getAttribute('aria-expanded') === 'true';
        attachToggle.setAttribute('aria-expanded', !isExpanded);
        uploadMenu.setAttribute('aria-hidden', isExpanded);
    });

    // 3. Close menu when clicking outside
    document.addEventListener('click', function(event) {
        if (!uploadMenu.contains(event.target) && !attachToggle.contains(event.target)) {
            attachToggle.setAttribute('aria-expanded', 'false');
            uploadMenu.setAttribute('aria-hidden', 'true');
        }
    });

    // 4. Handle file selection
    uploadFileInput.addEventListener('change', async function() {
        // Hide menu after selection
        attachToggle.setAttribute('aria-expanded', 'false');
        uploadMenu.setAttribute('aria-hidden', 'true');
        
        const file = this.files[0];
        if (!file) {
            return;
        }

        appendMessage('user', `Uploading file: <strong>${file.name}</strong>...`);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', { // We'll create this /upload route in Flask
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                appendMessage('ai', `File "${file.name}" uploaded successfully! Server response: ${data.message}`);
                // You might want to process this image on the backend to extract text for XAI
            } else {
                appendMessage('ai', `Error uploading file: ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            appendMessage('ai', 'Failed to upload file due to network or server issue.');
        }

        this.value = ''; // Clear the input so the same file can be selected again
    });
    
    // 5. Handle "Take Photo" button click
    takePhotoButton.addEventListener('click', function() {
        // Hide menu
        attachToggle.setAttribute('aria-expanded', 'false');
        uploadMenu.setAttribute('aria-hidden', 'true');
        
        // This would ideally open a modal with a video stream
        appendMessage('ai', 'Taking a photo feature is not fully implemented yet. Please use "Upload File" for now.');
        // TO-DO: Implement camera capture logic here (requires opening a video feed and capturing an image)
        // For a full implementation, you'd need a video element, stream handling, and a canvas to capture the image.
    });
});