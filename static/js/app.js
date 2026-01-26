// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const processBtn = document.getElementById('processBtn');
    const langSelect = document.getElementById('langSelect');
    const modelSelect = document.getElementById('modelSelect');
    const fileInfo = document.getElementById('fileInfo');
    const resultsSection = document.getElementById('resultsSection');
    const resultText = document.getElementById('resultText');
    const resultFilename = document.getElementById('resultFilename');
    const resultLang = document.getElementById('resultLang');
    const downloadBtn = document.getElementById('downloadBtn');
    const historyList = document.getElementById('historyList');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');

    // Transcribe Tab Elements
    const transcribeRecordBtn = document.getElementById('transcribeRecordBtn');
    const transcribeStopBtn = document.getElementById('transcribeStopBtn');
    const transcribeRecordingStatus = document.getElementById('transcribeRecordingStatus');
    const transcribeRecordingTimer = document.getElementById('transcribeRecordingTimer');
    const transcribeRecordingWarning = document.getElementById('transcribeRecordingWarning');
    const transcribeAudioVisualizer = document.getElementById('transcribeAudioVisualizer');
    const transcribeWaveformCanvas = document.getElementById('transcribeWaveformCanvas');

    // Contribute Tab Elements
    // Recording elements (Transcription Tab) - Removed/Replaced by Data Collection
    // const recordBtn = document.getElementById('recordBtn'); 
    // ...

    // Data Collection Elements
    const recordBtn = document.getElementById('recordBtn');
    const stopBtn = document.getElementById('stopBtn');
    const playBtn = document.getElementById('playBtn');
    const submitRecordingBtn = document.getElementById('submitRecordingBtn');
    const skipBtn = document.getElementById('skipBtn');

    // Mode Toggles
    const modePresetBtn = document.getElementById('modePresetBtn');
    const modeCustomBtn = document.getElementById('modeCustomBtn');
    const customPromptInput = document.getElementById('customPromptInput');
    const progressIndicator = document.getElementById('progressIndicator');
    const instructionText = document.getElementById('instructionText');

    const promptText = document.getElementById('promptText');
    const promptCategory = document.getElementById('promptCategory');
    const currentPromptIndexSpan = document.getElementById('currentPromptIndex');
    const totalPromptsSpan = document.getElementById('totalPrompts');
    const contributionMessage = document.getElementById('contributionMessage');
    const audioPlayback = document.getElementById('audioPlayback');
    const contributionList = document.getElementById('contributionList');

    // Visualizer
    const visualizer = document.getElementById('visualizer');

    // Tab Elements
    const tabs = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');

    // Check if all required transcription elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn) {
        console.error('Required Transcription DOM elements not found!');
        return;
    }

    // Check if all required elements exist
    if (!uploadArea || !fileInput || !uploadBtn || !processBtn || !recordBtn || !stopBtn) {
        console.error('Required DOM elements not found!');
        return;
    }

    // Global state variables
    let currentFile = null;
    let currentTranscription = null;
    let currentHistoryId = null;

    // Recording state
    let mediaRecorder = null;
    let audioChunks = [];
    let audioStream = null;
    let recordingBlob = null;

    // Data Collection State
    let prompts = [];
    let currentPromptIndex = 0;
    let isCustomMode = false;

    // Navigation Logic
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            const tabId = tab.dataset.tab;

            if (tabId === 'transcribe') {
                document.getElementById('transcribeTab').classList.add('active');
                fetchModels(); // Refresh models when switching to transcribe
                checkDiarizationAvailability(); // Re-check diarization availability
            } else if (tabId === 'long_audio') {
                document.getElementById('longAudioTab').classList.add('active');
                fetchModels(); // Refresh models when switching to long audio
            } else if (tabId === 'settings') {
                document.getElementById('settingsTab').classList.add('active');
                loadDiarizationSettings(); // Refresh diarization status
            } else {
                document.getElementById('contributeTab').classList.add('active');
                if (prompts.length === 0) loadPrompts();
            }
        });
    });

    // --- Long Audio Elements ---
    const longAudioUploadArea = document.getElementById('longAudioUploadArea');
    const longAudioFileInput = document.getElementById('longAudioFileInput');
    const longAudioUploadBtn = document.getElementById('longAudioUploadBtn');
    const longAudioProcessBtn = document.getElementById('longAudioProcessBtn');
    const longAudioLangSelect = document.getElementById('longAudioLangSelect');
    const longAudioModelSelect = document.getElementById('longAudioModelSelect');
    const longAudioFileInfo = document.getElementById('longAudioFileInfo');
    const longAudioResultsSection = document.getElementById('longAudioResultsSection');
    const longAudioResultsTable = document.getElementById('longAudioResultsTable');
    const longAudioResultFilename = document.getElementById('longAudioResultFilename');
    const longAudioResultLang = document.getElementById('longAudioResultLang');
    const longAudioDownloadBtn = document.getElementById('longAudioDownloadBtn');

    let currentLongAudioFile = null;
    let currentLongAudioResults = [];

    // Initialize Long Audio Model Select (Sync with main model select logic)
    // We'll reuse the fetchModels function but populate both selects if they exist
    // Modified fetchModels below to handle multiple selects or just call it again/copy options.
    // For simplicity, let's just make sure both get populated.

    if (longAudioModelSelect) {
        longAudioModelSelect.addEventListener('change', handleModelSwitch);
    }

    // Long Audio Event Listeners
    if (longAudioUploadBtn && longAudioFileInput) {
        longAudioUploadBtn.addEventListener('click', () => longAudioFileInput.click());
        longAudioFileInput.addEventListener('change', handleLongAudioFileSelect);

        longAudioUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            longAudioUploadArea.classList.add('dragover');
        });
        longAudioUploadArea.addEventListener('dragleave', () => {
            longAudioUploadArea.classList.remove('dragover');
        });
        longAudioUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            longAudioUploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) handleLongAudioFile(file);
        });
    }

    if (longAudioProcessBtn) {
        longAudioProcessBtn.addEventListener('click', handleLongAudioTranscribe);
    }

    if (longAudioDownloadBtn) {
        longAudioDownloadBtn.addEventListener('click', handleLongAudioDownload);
    }

    function handleLongAudioFileSelect(e) {
        const file = e.target.files[0];
        if (file) handleLongAudioFile(file);
    }

    function handleLongAudioFile(file) {
        if (!file.type.startsWith('audio/')) {
            alert('Please select an audio file');
            return;
        }
        currentLongAudioFile = file;
        longAudioFileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        longAudioProcessBtn.disabled = false;
        longAudioResultsSection.style.display = 'none';
    }

    async function handleLongAudioTranscribe() {
        if (!currentLongAudioFile) return;

        longAudioProcessBtn.disabled = true;
        const btnText = longAudioProcessBtn.querySelector('.btn-text');
        const btnSpinner = longAudioProcessBtn.querySelector('.btn-spinner');
        btnText.style.display = 'none';
        btnSpinner.style.display = 'inline';

        const formData = new FormData();
        formData.append('file', currentLongAudioFile);
        formData.append('lang_code', longAudioLangSelect.value);

        try {
            const response = await fetch('/api/transcribe_long', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Transcription failed');

            currentLongAudioResults = data.results; // List of {segment, filename, text}

            // Render Results
            renderLongAudioResults(data.results);

            longAudioResultFilename.textContent = `📄 ${data.filename}`;
            longAudioResultLang.textContent = `🌐 ${data.lang_code}`;
            longAudioResultsSection.style.display = 'block';
            longAudioResultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            longAudioProcessBtn.disabled = false;
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
        }
    }

    function renderLongAudioResults(results) {
        if (!results || results.length === 0) {
            longAudioResultsTable.innerHTML = '<p>No results found.</p>';
            return;
        }

        let html = '<div class="table-container" style="overflow-x:auto;"><table style="width:100%; border-collapse: collapse; margin-top: 1rem;">';
        html += '<thead><tr style="background:#f1f5f9; text-align:left;"><th style="padding:0.75rem; border-bottom:1px solid #e2e8f0;">Segment</th><th style="padding:0.75rem; border-bottom:1px solid #e2e8f0;">Text</th></tr></thead><tbody>';

        results.forEach(item => {
            html += `<tr style="border-bottom:1px solid #e2e8f0;">
                <td style="padding:0.75rem; white-space:nowrap; vertical-align:top; font-weight:bold; color:#64748b;">${item.segment}</td>
                <td style="padding:0.75rem;">${item.text}</td>
            </tr>`;
        });

        html += '</tbody></table></div>';
        longAudioResultsTable.innerHTML = html;
    }

    function handleLongAudioDownload() {
        if (!currentLongAudioResults || currentLongAudioResults.length === 0) return;

        const fullText = currentLongAudioResults.map(r => `[Segment ${r.segment}] ${r.text}`).join('\n\n');
        const blob = new Blob([fullText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `long_transcription_${currentLongAudioFile?.name || 'audio'}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Model switching event
    if (modelSelect) {
        modelSelect.addEventListener('change', handleModelSwitch);
    }

    // Function definitions (must be before event listeners)
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('audio/')) {
            alert('Please select an audio file');
            return;
        }

        currentFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        processBtn.disabled = false;
        resultsSection.style.display = 'none';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async function handleTranscribe() {
        if (!currentFile) {
            alert('Please select a file first');
            return;
        }

        // Check if diarization is enabled
        const useDiarization = enableDiarization && enableDiarization.checked && diarizationAvailable;

        // Disable button and show loading
        processBtn.disabled = true;
        const btnText = processBtn.querySelector('.btn-text');
        const btnSpinner = processBtn.querySelector('.btn-spinner');
        btnText.style.display = 'none';
        btnSpinner.style.display = 'inline';

        if (useDiarization) {
            // Use SSE streaming endpoint for diarization
            await handleDiarizedTranscribe();
        } else {
            // Use standard endpoint
            await handleStandardTranscribe();
        }
    }

    async function handleStandardTranscribe() {
        const btnText = processBtn.querySelector('.btn-text');
        const btnSpinner = processBtn.querySelector('.btn-spinner');

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('lang_code', langSelect.value);

        try {
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Transcription failed');
            }

            // Display results
            currentTranscription = data.transcription;
            currentHistoryId = data.history_id !== undefined ? data.history_id : null;
            resultText.textContent = data.transcription;
            resultFilename.textContent = `📄 ${data.filename}`;
            resultLang.textContent = `🌐 ${data.lang_code}`;
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            // Reload history
            loadHistory();

        } catch (error) {
            alert('Error: ' + error.message);
            console.error('Transcription error:', error);
        } finally {
            // Re-enable button
            processBtn.disabled = false;
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
        }
    }

    async function handleDiarizedTranscribe() {
        const btnText = processBtn.querySelector('.btn-text');
        const btnSpinner = processBtn.querySelector('.btn-spinner');

        // Get speaker count if specified
        const numSpeakers = speakerCountInput && speakerCountInput.value ? parseInt(speakerCountInput.value, 10) : null;

        // Prepare form data
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('lang_code', langSelect.value);
        if (numSpeakers && numSpeakers >= 1 && numSpeakers <= 10) {
            formData.append('num_speakers', numSpeakers);
        }

        // Reset progress and error states
        showProgress(true);
        hideSSEError();
        updateProgress('Initializing...', 0);
        currentDiarizationRetryCount = 0;
        speakerLabelMap = {}; // Reset speaker labels for new transcription
        currentSegments = [];

        try {
            // Submit file and get SSE stream
            const response = await fetch('/api/transcribe_with_diarization', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Diarization failed');
            }

            // For SSE streaming, the response is text/event-stream
            // We need to handle it with EventSource or read the stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let segments = [];

            // Show results section and prepare for streaming
            resultFilename.textContent = `📄 ${currentFile.name}`;
            resultLang.textContent = `🌐 ${langSelect.value}`;
            resultText.textContent = '';
            resultsSection.style.display = 'block';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Parse SSE events from buffer
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const eventData = JSON.parse(line.slice(6));
                            handleDiarizationEvent(eventData, segments);
                        } catch (e) {
                            console.warn('Failed to parse SSE event:', line);
                        }
                    }
                }
            }

            // Process any remaining buffer
            if (buffer.startsWith('data: ')) {
                try {
                    const eventData = JSON.parse(buffer.slice(6));
                    handleDiarizationEvent(eventData, segments);
                } catch (e) {
                    // Ignore incomplete data
                }
            }

            // Reload history
            loadHistory();

        } catch (error) {
            console.error('Diarization error:', error);

            // Handle SSE connection errors with retry option
            if (error.name === 'TypeError' || error.message.includes('network') || error.message.includes('fetch')) {
                showSSEError('Connection lost. Please check your network and try again.');
            } else {
                showSSEError(error.message);
            }

            showProgress(false);
        } finally {
            // Re-enable button
            processBtn.disabled = false;
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
        }
    }

    function showProgress(show) {
        if (diarizationProgress) {
            diarizationProgress.style.display = show ? 'block' : 'none';
        }
    }

    function updateProgress(stage, percent, details = '') {
        if (progressStage) {
            progressStage.textContent = stage;
        }
        if (progressPercent) {
            progressPercent.textContent = `${Math.round(percent)}%`;
        }
        if (progressBarFill) {
            progressBarFill.style.width = `${percent}%`;
            if (percent < 100) {
                progressBarFill.classList.add('animating');
            } else {
                progressBarFill.classList.remove('animating');
            }
        }
        if (progressDetails) {
            progressDetails.textContent = details;
        }
    }

    function showSSEError(message) {
        if (sseError) {
            sseError.style.display = 'flex';
        }
        if (sseErrorMessage) {
            sseErrorMessage.textContent = message;
        }
    }

    function hideSSEError() {
        if (sseError) {
            sseError.style.display = 'none';
        }
    }

    function handleDiarizationEvent(eventData, segments) {
        const eventType = eventData.event || eventData.type;

        switch (eventType) {
            case 'progress':
                // Update progress bar with stage-aware display
                const stage = eventData.stage || 'processing';
                const progress = eventData.progress || 0;
                const progressPercent = Math.round(progress * 100);

                // Map internal stages to user-friendly display
                let displayStage = 'Processing...';
                let details = '';

                if (stage === 'diarizing' || stage === 'diarization') {
                    displayStage = 'Identifying speakers...';
                    details = 'Analyzing audio to detect speaker turns';
                } else if (stage === 'transcribing' || stage === 'transcription') {
                    displayStage = 'Transcribing speech...';
                    const currentTurn = eventData.current_turn || 0;
                    const totalTurns = eventData.total_turns || 0;
                    if (totalTurns > 0) {
                        details = `Segment ${currentTurn} of ${totalTurns}`;
                    }
                } else if (stage === 'loading') {
                    displayStage = 'Loading models...';
                    details = 'Preparing diarization pipeline';
                } else if (stage === 'extracting') {
                    displayStage = 'Extracting audio...';
                    details = 'Preparing audio segments for transcription';
                } else if (stage === 'finalizing') {
                    displayStage = 'Finalizing...';
                    details = 'Assembling final transcript';
                }

                updateProgress(displayStage, progressPercent, details);
                break;

            case 'segment':
                // Add segment to results and update display
                segments.push(eventData);
                updateDiarizedDisplay(segments);
                break;

            case 'complete':
                // Final result received - hide progress bar
                showProgress(false);
                currentTranscription = eventData.full_transcript || segments.map(s => `[${s.speaker}] ${s.text}`).join('\n');
                currentHistoryId = eventData.history_id !== undefined ? eventData.history_id : null;
                updateDiarizedDisplay(segments);
                break;

            case 'error':
                // Show error in SSE error display
                showProgress(false);
                showSSEError(eventData.message || 'Unknown error occurred');
                break;
        }
    }

    function updateDiarizedDisplay(segments) {
        if (segments.length === 0) {
            resultText.textContent = 'Processing...';
            return;
        }

        // Store segments for re-rendering after label edits
        currentSegments = segments;

        // Build speaker-attributed transcript display with editable labels
        const html = segments.map((seg, index) => {
            const originalSpeaker = seg.speaker || 'Unknown';
            const displaySpeaker = speakerLabelMap[originalSpeaker] || originalSpeaker;
            const text = seg.text || '';
            const startTime = seg.start !== undefined ? formatTimestamp(seg.start) : '';
            const endTime = seg.end !== undefined ? formatTimestamp(seg.end) : '';
            const timeRange = startTime && endTime ? `[${startTime} - ${endTime}]` : '';
            return `<div class="diarization-segment" data-speaker-id="${escapeHtml(originalSpeaker)}">
                <span class="speaker-label-container">
                    <span class="speaker-label editable" data-original-speaker="${escapeHtml(originalSpeaker)}" title="Click to edit speaker name">${escapeHtml(displaySpeaker)}</span>
                    <button class="btn-edit-speaker" data-speaker="${escapeHtml(originalSpeaker)}" title="Rename speaker">&#9998;</button>
                </span>
                ${timeRange ? ` <span class="time-range">${timeRange}</span>` : ''}
                <br><span class="segment-text">${escapeHtml(text)}</span>
            </div>`;
        }).join('');

        resultText.innerHTML = html;

        // Update currentTranscription with custom labels
        currentTranscription = segments.map(s => {
            const displaySpeaker = speakerLabelMap[s.speaker] || s.speaker;
            return `[${displaySpeaker}] ${s.text}`;
        }).join('\n');

        // Add event listeners for edit buttons
        attachSpeakerEditListeners();
    }

    function attachSpeakerEditListeners() {
        document.querySelectorAll('.btn-edit-speaker').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const originalSpeaker = btn.dataset.speaker;
                openSpeakerEditDialog(originalSpeaker);
            });
        });

        // Also allow clicking on the speaker label itself
        document.querySelectorAll('.speaker-label.editable').forEach(label => {
            label.addEventListener('click', (e) => {
                e.stopPropagation();
                const originalSpeaker = label.dataset.originalSpeaker;
                openSpeakerEditDialog(originalSpeaker);
            });
        });
    }

    function openSpeakerEditDialog(originalSpeaker) {
        const currentName = speakerLabelMap[originalSpeaker] || originalSpeaker;
        const newName = prompt(`Rename speaker "${originalSpeaker}":\n(This will update all occurrences)`, currentName);

        if (newName !== null && newName.trim() !== '') {
            const trimmedName = newName.trim();
            if (trimmedName !== originalSpeaker) {
                speakerLabelMap[originalSpeaker] = trimmedName;
            } else {
                // If user sets it back to original, remove from map
                delete speakerLabelMap[originalSpeaker];
            }
            // Re-render with updated labels
            updateDiarizedDisplay(currentSegments);
        }
    }

    function resetSpeakerLabels() {
        speakerLabelMap = {};
        if (currentSegments.length > 0) {
            updateDiarizedDisplay(currentSegments);
        }
    }

    function formatTimestamp(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(1);
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(4, '0')}`;
    }

    function handleDownload() {
        if (!currentTranscription) {
            alert('No transcription to download');
            return;
        }

        // Get selected export format
        const exportFormatSelect = document.getElementById('exportFormatSelect');
        const format = exportFormatSelect ? exportFormatSelect.value : 'inline';

        // Find the history entry for current transcription
        if (currentHistoryId !== null) {
            // Include speaker renames in the export request
            let url = `/api/history/${currentHistoryId}/export?format=${format}`;
            if (Object.keys(speakerLabelMap).length > 0) {
                url += `&speaker_labels=${encodeURIComponent(JSON.stringify(speakerLabelMap))}`;
            }
            window.location.href = url;
        } else {
            // Create a temporary download for non-history transcriptions
            const blob = new Blob([currentTranscription], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const ext = format === 'json' ? '.json' : format === 'srt' ? '.srt' : '.txt';
            a.download = `transcription_${currentFile?.name || 'audio'}${ext}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }

    async function loadHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();

            if (data.history && data.history.length > 0) {
                historyList.innerHTML = data.history.map(item => createHistoryItem(item)).join('');

                // Add event listeners to history items
                document.querySelectorAll('.history-item').forEach(item => {
                    item.addEventListener('click', () => {
                        const historyId = parseInt(item.dataset.id);
                        loadHistoryItem(historyId);
                    });
                });

                document.querySelectorAll('.btn-download-history').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        const historyId = parseInt(btn.dataset.id);
                        window.location.href = `/api/download/${historyId}`;
                    });
                });
            } else {
                historyList.innerHTML = '<p class="empty-state">No history yet. Upload and transcribe an audio file to get started!</p>';
            }
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    function createHistoryItem(item) {
        const date = new Date(item.timestamp);
        const timeStr = date.toLocaleString();
        const preview = item.transcription.length > 150
            ? item.transcription.substring(0, 150) + '...'
            : item.transcription;

        return `
        <div class="history-item" data-id="${item.id}">
            <div class="history-item-header">
                <div class="history-item-meta">
                    <div class="history-item-filename">📄 ${item.filename}</div>
                    <div class="history-item-lang">🌐 ${item.lang_code}</div>
                    <div class="history-item-time">🕒 ${timeStr}</div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon btn-download-history" data-id="${item.id}" title="Download">
                        ⬇️
                    </button>
                </div>
            </div>
            <div class="history-item-text">${escapeHtml(preview)}</div>
        </div>
    `;
    }

    function loadHistoryItem(historyId) {
        // Load the full transcription from history
        fetch('/api/history')
            .then(response => response.json())
            .then(data => {
                const item = data.history.find(h => h.id === historyId);
                if (item) {
                    currentTranscription = item.transcription;
                    currentHistoryId = item.id;
                    resultText.textContent = item.transcription;
                    resultFilename.textContent = `📄 ${item.filename}`;
                    resultLang.textContent = `🌐 ${item.lang_code}`;
                    resultsSection.style.display = 'block';
                    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            })
            .catch(error => {
                console.error('Error loading history item:', error);
            });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Microphone Recording Functions
    async function startRecording() {
        try {
            // Request microphone access
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            // Set up audio context for visualization
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            const source = audioContext.createMediaStreamSource(audioStream);
            source.connect(analyser);
            dataArray = new Uint8Array(analyser.frequencyBinCount);

            // Set up MediaRecorder
            const options = { mimeType: 'audio/webm' };
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = 'audio/webm;codecs=opus';
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = ''; // Let browser choose
                }
            }

            mediaRecorder = new MediaRecorder(audioStream, options);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                await processRecordedAudio();
            };

            // Start recording
            mediaRecorder.start(100); // Collect data every 100ms
            recordingStartTime = Date.now();

            // Update UI
            recordBtn.disabled = true;
            recordBtn.style.display = 'none';
            stopBtn.disabled = false;
            stopBtn.style.display = 'inline-flex';
            recordingStatus.style.display = 'flex';
            audioVisualizer.style.display = 'block';
            recordingWarning.textContent = '';
            recordingWarning.className = 'recording-warning';

            // Start timer
            startTimer();

            // Start visualization
            visualizeAudio();

        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please check permissions and try again.');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }

        // Stop audio stream
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }

        // Stop timer and visualization
        stopTimer();
        stopVisualization();

        // Update UI
        recordBtn.disabled = false;
        recordBtn.style.display = 'inline-flex';
        stopBtn.disabled = true;
        stopBtn.style.display = 'none';
        recordingStatus.style.display = 'none';
        audioVisualizer.style.display = 'none';
    }

    function startTimer() {
        recordingStartTime = Date.now();
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            recordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

            // Warn at 35 seconds, stop at 40 seconds
            if (elapsed >= MAX_RECORDING_TIME) {
                stopRecording();
                alert(`Recording stopped automatically at ${MAX_RECORDING_TIME} seconds (maximum length).`);
            } else if (elapsed >= 35) {
                recordingWarning.textContent = `⚠️ Recording will stop automatically at ${MAX_RECORDING_TIME} seconds`;
                recordingWarning.className = 'recording-warning warning';
            }
        }, 100);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function visualizeAudio() {
        if (!analyser || !waveformCanvas) return;

        const canvas = waveformCanvas;
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        function draw() {
            if (!analyser) return;

            animationFrameId = requestAnimationFrame(draw);

            analyser.getByteFrequencyData(dataArray);

            ctx.fillStyle = '#f8fafc';
            ctx.fillRect(0, 0, width, height);

            const barWidth = width / dataArray.length * 2.5;
            let x = 0;

            for (let i = 0; i < dataArray.length; i++) {
                const barHeight = (dataArray[i] / 255) * height * 0.8;
                const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
                gradient.addColorStop(0, '#6366f1');
                gradient.addColorStop(1, '#8b5cf6');

                ctx.fillStyle = gradient;
                ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }

        draw();
    }

    function stopVisualization() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (waveformCanvas) {
            const ctx = waveformCanvas.getContext('2d');
            ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    }

    // Data Collection Functions
    async function loadPrompts() {
        try {
            promptText.textContent = "Loading prompts...";
            const response = await fetch('/api/prompts');
            const data = await response.json();
            if (data.prompts) {
                prompts = data.prompts;
                totalPromptsSpan.textContent = prompts.length;
                loadPrompt(0);
            }
        } catch (e) {
            console.error("Failed to load prompts", e);
            promptText.textContent = "Error loading prompts.";
        }
    }

    function loadPrompt(index) {
        if (index >= prompts.length) {
            promptText.textContent = "All prompts completed! Thank you!";
            promptCategory.textContent = "Complete";
            recordBtn.disabled = true;
            return;
        }
        currentPromptIndex = index;
        const prompt = prompts[index];
        promptText.textContent = prompt.text;
        promptCategory.textContent = prompt.category; // + " (" + prompt.lang_code + ")"
        currentPromptIndexSpan.textContent = index + 1;

        // Reset UI
        resetRecordingUI();
    }

    function resetRecordingUI() {
        recordingBlob = null;
        audioChunks = [];
        audioPlayback.src = "";

        recordBtn.disabled = false;
        recordBtn.style.display = 'inline-flex';
        stopBtn.disabled = true;
        stopBtn.style.display = 'none';
        playBtn.disabled = true;
        submitRecordingBtn.disabled = true;

        visualizer.classList.remove('recording');
        document.querySelector('.recording-status-text').textContent = "Ready to Record";
    }

    /* --- Model Switching Logic --- */

    async function fetchModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            if (data.models) {
                [modelSelect, longAudioModelSelect].forEach(select => {
                    if (!select) return;
                    select.innerHTML = '';
                    Object.entries(data.models).forEach(([displayName, modelCard]) => {
                        const option = document.createElement('option');
                        option.value = modelCard;
                        option.textContent = displayName;
                        if (modelCard === data.current_model) {
                            option.selected = true;
                        }
                        select.appendChild(option);
                    });
                });
            }
        } catch (e) {
            console.error("Failed to load models", e);
            [modelSelect, longAudioModelSelect].forEach(select => {
                if (select) select.innerHTML = '<option disabled>Error loading models</option>';
            });
        }
    }

    async function handleModelSwitch(e) {
        const selectedModel = e.target.value;
        const previousModel = e.target.getAttribute('data-prev') || selectedModel; // fallback

        // Confirmation or direct switch? Let's just switch with UI feedback
        const confirmSwitch = confirm("Switching models will take a few seconds and clear current memory. Continue?");
        if (!confirmSwitch) {
            e.target.value = previousModel; // Revert
            return;
        }

        // Disable UI
        document.body.style.cursor = 'wait';
        modelSelect.disabled = true;
        processBtn.disabled = true;

        // Store current as previous for next time
        e.target.setAttribute('data-prev', selectedModel);

        try {
            const response = await fetch('/api/model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_card: selectedModel })
            });
            const data = await response.json();

            if (data.success) {
                alert(`Successfully switched to model: ${selectedModel}`);
            } else {
                throw new Error(data.error || "Unknown error switching model");
            }
        } catch (error) {
            alert(`Error switching model: ${error.message}`);
            // Revert selection if possible, though strict sync with backend might differ
            // We'll just fetch active model again to be sure
            fetchModels();
        } finally {
            document.body.style.cursor = 'default';
            modelSelect.disabled = false;
            if (currentFile) processBtn.disabled = false;
        }
    }

    /* --- Mode Switching Logic --- */

    function setCollectionMode(mode) {
        isCustomMode = (mode === 'custom');

        // Update Buttons
        if (isCustomMode) {
            modePresetBtn.classList.remove('active');
            modeCustomBtn.classList.add('active');

            // UI Changes
            promptText.style.display = 'none';
            customPromptInput.style.display = 'block';
            progressIndicator.style.display = 'none';
            skipBtn.style.display = 'none';
            instructionText.innerText = "Type your text, then record.";
            promptCategory.textContent = "Custom Input";

            // Reset for new custom input
            resetRecordingUI();

        } else {
            modePresetBtn.classList.add('active');
            modeCustomBtn.classList.remove('active');

            // UI Changes
            promptText.style.display = 'block';
            customPromptInput.style.display = 'none';
            progressIndicator.style.display = 'inline-block';
            skipBtn.style.display = 'inline-block';
            instructionText.innerText = "Read the prompt below and record your voice.";

            // Reload current preset
            loadPrompt(currentPromptIndex);
        }
    }

    /* --- Contribute Recording Logic --- */

    async function startContributeRecording() {
        if (isCustomMode && !customPromptInput.value.trim()) {
            alert("Please type some text first!");
            return;
        }

        try {
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, sampleRate: 16000 }
            });
            mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
            audioChunks = [];
            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
            mediaRecorder.onstop = () => {
                recordingBlob = new Blob(audioChunks, { type: 'audio/webm' });
                audioPlayback.src = URL.createObjectURL(recordingBlob);
                playBtn.disabled = false;
                submitRecordingBtn.disabled = false;
                visualizer.classList.remove('recording');
                document.querySelector('.recording-status-text').textContent = "Recording Stopped";
            };
            mediaRecorder.start();

            recordBtn.style.display = 'none';
            stopBtn.style.display = 'inline-flex';
            stopBtn.disabled = false;
            visualizer.classList.add('recording');
            document.querySelector('.recording-status-text').textContent = "Recording...";

            // Disable input while recording
            if (isCustomMode) customPromptInput.disabled = true;

        } catch (e) {
            alert("Microphone error: " + e.message);
        }
    }

    function stopContributeRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
        if (audioStream) audioStream.getTracks().forEach(t => t.stop());
        if (isCustomMode) customPromptInput.disabled = false;
    }

    function playRecording() {
        if (audioPlayback.src) {
            audioPlayback.play();
        }
    }

    async function submitRecording() {
        if (!recordingBlob) return;

        let id, text, lang;

        if (isCustomMode) {
            text = customPromptInput.value.trim();
            id = "custom_" + Date.now();
            lang = "ben_Beng"; // Default to Bangla for custom, or could add selector
            if (!text) return alert("Text required!");
            customPromptInput.value = ""; // Clear after submit
        } else {
            const prompt = prompts[currentPromptIndex];
            text = prompt.text;
            id = prompt.id;
            lang = prompt.lang_code;
        }

        const formData = new FormData();
        formData.append("audio", recordingBlob, "recording.webm");
        formData.append("prompt_id", id);
        formData.append("transcript", text);
        formData.append("lang_code", lang);

        submitRecordingBtn.disabled = true;
        submitRecordingBtn.textContent = "Submitting...";

        try {
            const response = await fetch('/api/dataset/submit', { method: 'POST', body: formData });
            const result = await response.json();

            if (result.success) {
                contributionMessage.textContent = "Saved successfully!";
                contributionMessage.className = "contribution-message success";
                addToContributionList(text, lang);

                // Only advance if in preset mode
                if (!isCustomMode) {
                    setTimeout(() => {
                        loadPrompt(currentPromptIndex + 1);
                    }, 1000);
                } else {
                    resetRecordingUI(); // Just reset for next custom input
                }
            } else {
                throw new Error(result.error);
            }
        } catch (e) {
            contributionMessage.textContent = "Error: " + e.message;
            contributionMessage.className = "contribution-message error";
        } finally {
            submitRecordingBtn.disabled = false;
            submitRecordingBtn.textContent = "Submit & Next";
        }
    }

    function skipPrompt() {
        loadPrompt(currentPromptIndex + 1);
    }

    function addToContributionList(text, lang) {
        const div = document.createElement('div');
        div.className = "history-item";
        div.style.padding = "1rem";
        div.innerHTML = `<div class="history-item-text">✅ ${text} <small>(${lang})</small></div>`;
        contributionList.prepend(div);

        const emptyState = contributionList.querySelector('.empty-state');
        if (emptyState) emptyState.remove();
    }

    // Convert AudioBuffer to WAV format
    function audioBufferToWav(buffer) {
        const numChannels = buffer.numberOfChannels;
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;

        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;

        const length = buffer.length * numChannels * bytesPerSample;
        const arrayBuffer = new ArrayBuffer(44 + length);
        const view = new DataView(arrayBuffer);

        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, length, true);

        // Convert audio data
        let offset = 44;
        for (let i = 0; i < buffer.length; i++) {
            for (let channel = 0; channel < numChannels; channel++) {
                const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                offset += 2;
            }
        }

        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }

    // Event Listeners (after function definitions)
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    /* --- Transcription Recording Logic --- */

    async function startTranscribeRecording() {
        try {
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true }
            });

            // Visualizer setup
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            const source = audioContext.createMediaStreamSource(audioStream);
            source.connect(analyser);
            dataArray = new Uint8Array(analyser.frequencyBinCount);

            mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                await processTranscribeRecording();
            };

            mediaRecorder.start(100);
            recordingStartTime = Date.now();

            // UI Updates
            transcribeRecordBtn.style.display = 'none';
            transcribeStopBtn.style.display = 'inline-flex';
            transcribeStopBtn.disabled = false;
            transcribeRecordingStatus.style.display = 'flex';
            transcribeAudioVisualizer.style.display = 'block';

            startTranscribeTimer();
            visualizeTranscribeAudio();

        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone.');
        }
    }

    function stopTranscribeRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            // Don't nullify yet if we want to reuse? No, better get fresh stream next time.
        }
        stopTranscribeTimer();
        stopTranscribeVisualization();

        // UI Reset
        transcribeRecordBtn.style.display = 'inline-flex';
        transcribeStopBtn.style.display = 'none';
        transcribeRecordingStatus.style.display = 'none';
        transcribeAudioVisualizer.style.display = 'none';
    }

    function startTranscribeTimer() {
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            transcribeRecordingTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
            if (elapsed >= 40) stopTranscribeRecording();
        }, 100);
    }

    function stopTranscribeTimer() {
        clearInterval(timerInterval);
    }

    function visualizeTranscribeAudio() {
        if (!analyser || !transcribeWaveformCanvas) return;
        const ctx = transcribeWaveformCanvas.getContext('2d');
        const width = transcribeWaveformCanvas.width;
        const height = transcribeWaveformCanvas.height;

        function draw() {
            if (!analyser) return;
            animationFrameId = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);
            ctx.fillStyle = '#f8fafc';
            ctx.fillRect(0, 0, width, height);

            const barWidth = width / dataArray.length * 2.5;
            let x = 0;
            for (let i = 0; i < dataArray.length; i++) {
                const barHeight = (dataArray[i] / 255) * height * 0.8;
                const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
                gradient.addColorStop(0, '#6366f1');
                gradient.addColorStop(1, '#8b5cf6');
                ctx.fillStyle = gradient;
                ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }
        draw();
    }

    function stopTranscribeVisualization() {
        cancelAnimationFrame(animationFrameId);
        if (audioContext) audioContext.close();
    }

    async function processTranscribeRecording() {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        try {
            const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
            const ab = await audioBlob.arrayBuffer();
            const audioBuffer = await decodeCtx.decodeAudioData(ab);
            const wavBlob = audioBufferToWav(audioBuffer);
            await decodeCtx.close();

            const wavFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
            currentFile = wavFile;
            fileInfo.textContent = `Recorded: ${wavFile.name} (${formatFileSize(wavFile.size)})`;
            processBtn.disabled = false;

            // Auto-transcribe
            await handleTranscribe();
        } catch (e) {
            console.error(e);
            alert("Error processing recording.");
        }
    }

    /* --- Contribute Recording Logic (Simplified) --- */

    async function startContributeRecording() {
        try {
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, sampleRate: 16000 }
            });
            mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
            audioChunks = [];
            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
            mediaRecorder.onstop = () => {
                recordingBlob = new Blob(audioChunks, { type: 'audio/webm' });
                audioPlayback.src = URL.createObjectURL(recordingBlob);
                playBtn.disabled = false;
                submitRecordingBtn.disabled = false;
                visualizer.classList.remove('recording');
                document.querySelector('.recording-status-text').textContent = "Recording Stopped";
            };
            mediaRecorder.start();

            recordBtn.style.display = 'none';
            stopBtn.style.display = 'inline-flex';
            stopBtn.disabled = false;
            visualizer.classList.add('recording');
            document.querySelector('.recording-status-text').textContent = "Recording...";
        } catch (e) {
            alert("Microphone error: " + e.message);
        }
    }

    function stopContributeRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
        if (audioStream) audioStream.getTracks().forEach(t => t.stop());
    }

    // Event Listeners
    processBtn.addEventListener('click', handleTranscribe);
    downloadBtn.addEventListener('click', handleDownload);
    refreshHistoryBtn.addEventListener('click', loadHistory);

    // Transcribe Recording Events
    if (transcribeRecordBtn) {
        transcribeRecordBtn.addEventListener('click', startTranscribeRecording);
        transcribeStopBtn.addEventListener('click', stopTranscribeRecording);
    }

    // Contribute Recording Events
    recordBtn.addEventListener('click', startContributeRecording);
    stopBtn.addEventListener('click', stopContributeRecording);
    playBtn.addEventListener('click', playRecording);
    submitRecordingBtn.addEventListener('click', submitRecording);
    skipBtn.addEventListener('click', skipPrompt);

    modePresetBtn.addEventListener('click', () => setCollectionMode('preset'));
    modeCustomBtn.addEventListener('click', () => setCollectionMode('custom'));

    /* --- Diarization Options Management --- */

    // Diarization UI Elements
    const diarizationOptions = document.getElementById('diarizationOptions');
    const diarizationNotice = document.getElementById('diarizationNotice');
    const enableDiarization = document.getElementById('enableDiarization');
    const speakerCountRow = document.getElementById('speakerCountRow');
    const speakerCountInput = document.getElementById('speakerCount');
    const enableDiarizationLink = document.getElementById('enableDiarizationLink');

    // Progress Bar Elements
    const diarizationProgress = document.getElementById('diarizationProgress');
    const progressStage = document.getElementById('progressStage');
    const progressPercent = document.getElementById('progressPercent');
    const progressBarFill = document.getElementById('progressBarFill');
    const progressDetails = document.getElementById('progressDetails');

    // SSE Error Elements
    const sseError = document.getElementById('sseError');
    const sseErrorMessage = document.getElementById('sseErrorMessage');
    const sseRetryBtn = document.getElementById('sseRetryBtn');

    // Diarization state
    let diarizationAvailable = false;
    let currentDiarizationRetryCount = 0;
    const MAX_SSE_RETRIES = 3;
    let speakerLabelMap = {}; // Maps original speaker ID (e.g., SPEAKER_00) to custom name
    let currentSegments = []; // Store current segments for re-rendering after label edits

    async function checkDiarizationAvailability() {
        try {
            const response = await fetch('/api/settings/diarization');
            const data = await response.json();

            diarizationAvailable = data.diarization_available === true;

            if (diarizationAvailable) {
                // Show diarization options
                if (diarizationOptions) diarizationOptions.style.display = 'block';
                if (diarizationNotice) diarizationNotice.style.display = 'none';
            } else if (data.hf_token_set) {
                // Token set but not available (license not accepted)
                if (diarizationOptions) diarizationOptions.style.display = 'none';
                if (diarizationNotice) {
                    diarizationNotice.style.display = 'none'; // Don't show notice if token issues
                }
            } else {
                // No token set
                if (diarizationOptions) diarizationOptions.style.display = 'none';
                if (diarizationNotice) diarizationNotice.style.display = 'none';
            }
        } catch (e) {
            console.error('Failed to check diarization availability', e);
            if (diarizationOptions) diarizationOptions.style.display = 'none';
            if (diarizationNotice) diarizationNotice.style.display = 'none';
        }
    }

    function toggleSpeakerCountVisibility() {
        if (!enableDiarization || !speakerCountRow) return;

        if (enableDiarization.checked) {
            speakerCountRow.style.display = 'flex';
        } else {
            speakerCountRow.style.display = 'none';
            if (speakerCountInput) speakerCountInput.value = '';
        }
    }

    // Event listeners for diarization options
    if (enableDiarization) {
        enableDiarization.addEventListener('change', toggleSpeakerCountVisibility);
    }

    if (enableDiarizationLink) {
        enableDiarizationLink.addEventListener('click', (e) => {
            e.preventDefault();
            if (enableDiarization) {
                enableDiarization.checked = true;
                toggleSpeakerCountVisibility();
            }
        });
    }

    // SSE Retry Button
    if (sseRetryBtn) {
        sseRetryBtn.addEventListener('click', (e) => {
            e.preventDefault();
            currentDiarizationRetryCount++;
            if (currentDiarizationRetryCount <= MAX_SSE_RETRIES) {
                hideSSEError();
                handleTranscribe();
            } else {
                showSSEError('Maximum retry attempts reached. Please try again later.');
            }
        });
    }

    /* --- Settings Management --- */

    // Settings Elements
    const hfTokenInput = document.getElementById('hfTokenInput');
    const saveHfTokenBtn = document.getElementById('saveHfTokenBtn');
    const toggleTokenVisibility = document.getElementById('toggleTokenVisibility');
    const tokenSaveMessage = document.getElementById('tokenSaveMessage');
    const diarizationIndicator = document.getElementById('diarizationIndicator');

    async function loadDiarizationSettings() {
        if (!diarizationIndicator) return;

        try {
            const response = await fetch('/api/settings/diarization');
            const data = await response.json();

            updateDiarizationStatus(data);
        } catch (e) {
            console.error('Failed to load diarization settings', e);
            updateDiarizationStatus({ error: 'Failed to check status' });
        }
    }

    function updateDiarizationStatus(data) {
        if (!diarizationIndicator) return;

        const statusDot = diarizationIndicator.querySelector('.status-dot');
        const statusText = diarizationIndicator.querySelector('.status-text');

        if (data.error) {
            statusDot.className = 'status-dot error';
            statusText.textContent = 'Error: ' + data.error;
            return;
        }

        if (data.diarization_available) {
            statusDot.className = 'status-dot available';
            statusText.textContent = 'Diarization available';
        } else if (data.hf_token_set) {
            statusDot.className = 'status-dot pending';
            statusText.textContent = 'Token set, but model not available (check license agreements)';
        } else {
            statusDot.className = 'status-dot unavailable';
            statusText.textContent = 'Diarization unavailable (token not set)';
        }
    }

    async function saveHfToken() {
        if (!hfTokenInput || !tokenSaveMessage) return;

        const token = hfTokenInput.value.trim();

        if (!token) {
            showTokenMessage('Please enter a token', 'error');
            return;
        }

        saveHfTokenBtn.disabled = true;
        saveHfTokenBtn.textContent = 'Saving...';

        try {
            const response = await fetch('/api/settings/hf_token', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token: token })
            });

            const data = await response.json();

            if (data.success) {
                showTokenMessage('Token saved successfully!', 'success');
                hfTokenInput.value = ''; // Clear input for security
                loadDiarizationSettings(); // Refresh status
            } else {
                throw new Error(data.error || 'Failed to save token');
            }
        } catch (e) {
            showTokenMessage('Error: ' + e.message, 'error');
        } finally {
            saveHfTokenBtn.disabled = false;
            saveHfTokenBtn.textContent = 'Save Token';
        }
    }

    function showTokenMessage(message, type) {
        if (!tokenSaveMessage) return;

        tokenSaveMessage.textContent = message;
        tokenSaveMessage.className = 'settings-message ' + type;

        // Clear message after 5 seconds
        setTimeout(() => {
            tokenSaveMessage.textContent = '';
            tokenSaveMessage.className = 'settings-message';
        }, 5000);
    }

    function toggleTokenInputVisibility() {
        if (!hfTokenInput || !toggleTokenVisibility) return;

        if (hfTokenInput.type === 'password') {
            hfTokenInput.type = 'text';
            toggleTokenVisibility.textContent = '🙈';
        } else {
            hfTokenInput.type = 'password';
            toggleTokenVisibility.textContent = '👁️';
        }
    }

    // Settings Event Listeners
    if (saveHfTokenBtn) {
        saveHfTokenBtn.addEventListener('click', saveHfToken);
    }

    if (toggleTokenVisibility) {
        toggleTokenVisibility.addEventListener('click', toggleTokenInputVisibility);
    }

    if (hfTokenInput) {
        hfTokenInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                saveHfToken();
            }
        });
    }

    // Initialize
    fetchModels();
    loadHistory();
    checkDiarizationAvailability();
});
