import { API_BASE_URL, postJson, deleteRequest } from "./api";

// Verify the patient by full name and open a session.
// The backend resolves the internal patient_id itself and never returns it -
// every later request identifies the patient through the session_id alone.
export async function createSession(fullName) {
    return await postJson("/sessions", { full_name: fullName });
}

// Send a chat message. Returns { message_id, response, sources }.
export async function sendMessage(sessionId, message) {
    return await postJson(`/sessions/${sessionId}/chat`, { message: message });
}

// Delete the session and its conversation history
export async function deleteSession(sessionId) {
    return await deleteRequest(`/sessions/${sessionId}`);
}

// Fire-and-forget delete for page unload, where a normal awaited fetch would
// be cancelled before it leaves the browser.
export function deleteSessionOnPageExit(sessionId) {
    fetch(
        `${API_BASE_URL}/sessions/${sessionId}`,
        {
            method: "DELETE",
            keepalive: true,
        }
    );
}
