const API_BASE_URL =
    window.APP_CONFIG?.apiBaseUrl?.trim() ||
    `${window.location.protocol}//${window.location.hostname}:8000`;

const API_URL = `${API_BASE_URL}/ask`;
const HEALTH_URL = `${API_BASE_URL}/health`;

// --- DOM Elements ---
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatMessages = document.getElementById("chat-messages");
const citationsList = document.getElementById("citations-list");
const sendBtn = document.getElementById("send-btn");
const clearChatBtn = document.getElementById("clear-chat");
const systemStatus = document.getElementById("system-status");
const statusDot = document.querySelector(".status-dot");
const inputGlowWrapper = document.querySelector(".input-glow-wrapper");

// --- State ---
let isProcessing = false;
let healthIntervalId = null;
let activeTypingIndicator = null;

// --- Markdown Configuration ---
if (window.marked) {
    marked.setOptions({
        breaks: true,
        gfm: true
    });
}

// --- Helper Functions ---

function renderMarkdownSafe(markdownText) {
    const rawMarkdown = markdownText ?? "";
    const parsedHtml = window.marked ? marked.parse(rawMarkdown) : rawMarkdown;
    return window.DOMPurify
        ? DOMPurify.sanitize(parsedHtml)
        : parsedHtml;
}

// Tidy the inline [n] citation markers the model emits:
//  - merge an adjacent run like "[1][2]" into a single "[1, 2]" (also avoids
//    `marked` misreading "[1][2]" as reference-link syntax)
//  - collapse consecutive runs that cite the exact same set down to the last
//    one, so a marker isn't repeated after every sentence
// Collapsing stops at paragraph breaks, so each paragraph keeps its citation.
function normalizeCitations(text) {
    if (!text) return text;

    const runRegex = /\[\d+\](?!\()(?:\s*\[\d+\](?!\())*/g;
    const runs = [];
    let match;
    while ((match = runRegex.exec(text)) !== null) {
        const nums = [...new Set(match[0].match(/\d+/g).map(Number))].sort((a, b) => a - b);
        runs.push({ start: match.index, end: match.index + match[0].length, nums });
    }
    if (runs.length === 0) return text;

    const keep = runs.map(() => true);
    for (let i = 0; i < runs.length - 1; i++) {
        const a = runs[i];
        const b = runs[i + 1];
        const sameSet = a.nums.length === b.nums.length &&
            a.nums.every((n, idx) => n === b.nums[idx]);
        const crossesParagraph = /\n\s*\n/.test(text.slice(a.end, b.start));
        if (sameSet && !crossesParagraph) keep[i] = false;
    }

    let result = text;
    for (let i = runs.length - 1; i >= 0; i--) {
        const run = runs[i];
        const replacement = keep[i] ? `[${run.nums.join(", ")}]` : "";
        result = result.slice(0, run.start) + replacement + result.slice(run.end);
    }

    // Clean up gaps left where markers were removed.
    return result
        .replace(/ {2,}/g, " ")
        .replace(/ ([.,;:!?])/g, "$1")
        .replace(/[ \t]+\n/g, "\n");
}

function updateSendButtonState() {
    const hasText = userInput.value.trim().length > 0;
    sendBtn.disabled = isProcessing || !hasText;
}

function addMessage(text, role) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    if (role === "assistant") {
        contentDiv.innerHTML = renderMarkdownSafe(normalizeCitations(text));
    } else {
        contentDiv.textContent = text ?? "";
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const typingDiv = document.createElement("div");
    typingDiv.className = "message assistant typing-indicator";
    typingDiv.setAttribute("aria-label", "Assistant is typing");

    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;

    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typingDiv;
}

function removeTypingIndicator() {
    if (activeTypingIndicator) {
        activeTypingIndicator.remove();
        activeTypingIndicator = null;
    }
}

function renderEmptyCitations(message) {
    citationsList.replaceChildren();

    const emptyState = document.createElement("div");
    emptyState.className = "empty-state";

    const p = document.createElement("p");
    p.textContent = message;

    emptyState.appendChild(p);
    citationsList.appendChild(emptyState);
}

function createMetaRow(label, value) {
    const row = document.createElement("div");
    row.className = "citation-detail-row";

    const strong = document.createElement("strong");
    strong.textContent = `${label}: `;

    const span = document.createElement("span");
    span.textContent = value ?? "N/A";

    row.appendChild(strong);
    row.appendChild(span);
    return row;
}

function toggleCitationCard(card) {
    card.classList.toggle("expanded");
}

function updateCitations(citations) {
    if (!citations || citations.length === 0) {
        renderEmptyCitations("No specific citations for this response.");
        return;
    }

    citationsList.replaceChildren();

    citations.forEach((citation, index) => {
        const card = document.createElement("div");
        card.className = "citation-card";
        card.tabIndex = 0;
        card.setAttribute("role", "button");
        card.setAttribute("aria-expanded", "false");
        card.setAttribute("aria-label", `Citation ${index + 1}. Press Enter or Space to expand.`);

        const meta = document.createElement("div");
        meta.className = "meta";

        const sourceLabel = document.createElement("span");
        sourceLabel.textContent = `Citation #${index + 1}`;

        const expandLabel = document.createElement("span");
        expandLabel.textContent = "(Click to expand)";

        meta.appendChild(sourceLabel);
        meta.appendChild(expandLabel);

        const details = document.createElement("div");
        details.className = "citation-details";

        details.appendChild(createMetaRow("Source", citation.source_file || "Unknown Source"));
        details.appendChild(createMetaRow("Page", String(citation.page_number ?? "Unknown Page")));
        details.appendChild(createMetaRow("Section", citation.section_header || "Unknown Section"));
        details.appendChild(createMetaRow("Type", citation.content_type || "text"));

        const excerpt = document.createElement("div");
        excerpt.className = "excerpt";
        excerpt.textContent = citation.excerpt ?? "";

        card.appendChild(meta);
        card.appendChild(details);
        card.appendChild(excerpt);

        card.addEventListener("click", () => {
            toggleCitationCard(card);
            card.setAttribute("aria-expanded", card.classList.contains("expanded") ? "true" : "false");
        });

        card.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                toggleCitationCard(card);
                card.setAttribute("aria-expanded", card.classList.contains("expanded") ? "true" : "false");
            }
        });

        citationsList.appendChild(card);
    });
}

function setProcessing(bool) {
    isProcessing = bool;
    userInput.disabled = bool;

    if (bool) {
        inputGlowWrapper?.classList.add("loading");
        userInput.placeholder = "Brain is working...";
    } else {
        inputGlowWrapper?.classList.remove("loading");
        userInput.placeholder = "Ask about your research papers...";
    }

    updateSendButtonState();
}

function setSystemStatus(state) {
    if (!systemStatus || !statusDot) return;

    if (state === "checking") {
        systemStatus.textContent = "Checking...";
        statusDot.style.backgroundColor = "#d29922";
        statusDot.style.boxShadow = "0 0 8px rgba(210, 153, 34, 0.5)";
        return;
    }

    if (state === "online") {
        systemStatus.textContent = "Brain Online";
        statusDot.style.backgroundColor = "#3fb950";
        statusDot.style.boxShadow = "0 0 8px rgba(63, 185, 80, 0.5)";
        return;
    }

    systemStatus.textContent = "Brain Offline";
    statusDot.style.backgroundColor = "#f85149";
    statusDot.style.boxShadow = "0 0 8px rgba(248, 81, 73, 0.5)";
}

async function checkHealth() {
    setSystemStatus("checking");

    try {
        const response = await fetch(HEALTH_URL, {
            method: "GET",
            cache: "no-store"
        });

        if (!response.ok) {
            throw new Error(`Health check failed (${response.status})`);
        }

        const data = await response.json();

        if (data.status === "ok") {
            setSystemStatus("online");
        } else {
            setSystemStatus("offline");
        }
    } catch (error) {
        console.error("Health check error:", error);
        setSystemStatus("offline");
    }
}

function startHealthPolling() {
    checkHealth();

    if (healthIntervalId) {
        clearInterval(healthIntervalId);
    }

    healthIntervalId = setInterval(checkHealth, 10000);
}

async function parseErrorResponse(response) {
    try {
        const data = await response.json();
        return data?.detail || data?.message || `Request failed with status ${response.status}.`;
    } catch {
        return `Request failed with status ${response.status}.`;
    }
}

function getFriendlyErrorMessage(error, statusCode = null, detail = "") {
    if (statusCode === 400) {
        return detail || "The request was invalid. Please check your input.";
    }

    if (statusCode === 401 || statusCode === 403) {
        return "Access to the Brain API was denied.";
    }

    if (statusCode === 404) {
        return "The requested API endpoint was not found.";
    }

    if (statusCode === 422) {
        return detail || "The request format was invalid.";
    }

    if (statusCode >= 500) {
        return detail || "The Brain encountered an internal error while processing your question.";
    }

    if (error instanceof TypeError) {
        return "Could not reach the Brain API. Check that the backend is running and the API URL is correct.";
    }

    return detail || "An unexpected error occurred while processing your request.";
}

async function handleSubmit(e) {
    e.preventDefault();

    const query = userInput.value.trim();
    if (!query || isProcessing) return;

    addMessage(query, "user");
    userInput.value = "";
    setProcessing(true);
    activeTypingIndicator = showTypingIndicator();

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
        });

        if (!response.ok) {
            const detail = await parseErrorResponse(response);
            const message = getFriendlyErrorMessage(null, response.status, detail);
            throw new Error(message);
        }

        let data;
        try {
            data = await response.json();
        } catch {
            throw new Error("The Brain returned an invalid JSON response.");
        }

        removeTypingIndicator();
        addMessage(data.answer ?? "No answer was returned.", "assistant");
        updateCitations(data.citations || []);
        setSystemStatus("online");

    } catch (error) {
        console.error("Request error:", error);
        removeTypingIndicator();

        const friendlyMessage = getFriendlyErrorMessage(error);
        addMessage(`**Request failed**\n\n${friendlyMessage}`, "assistant");
        renderEmptyCitations("No citations available because the request failed.");

        if (error instanceof TypeError) {
            setSystemStatus("offline");
        }
    } finally {
        removeTypingIndicator();
        setProcessing(false);
        updateSendButtonState();
    }
}

// --- Event Listeners ---
chatForm.addEventListener("submit", handleSubmit);

userInput.addEventListener("input", updateSendButtonState);

clearChatBtn.addEventListener("click", () => {
    chatMessages.replaceChildren();
    renderEmptyCitations("Ask a question to see source citations here.");
    addMessage("Chat cleared. How else can I help you?", "assistant");
    updateSendButtonState();
});

// Initial setup
userInput.focus();
updateSendButtonState();
startHealthPolling();