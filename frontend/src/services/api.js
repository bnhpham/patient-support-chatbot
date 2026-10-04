//----------------------------------------
/* Connects to FastAPI backend */
//----------------------------------------

const API_BASE_URL = "http://localhost:8000";

// Turn a failed response into an Error carrying the backend's own message.
// The backend answers errors with {"detail": "..."}, so the UI can show
// "No matching patient found" instead of a generic failure.
async function toError(response) {
    let detail = "";

    try {
        const body = await response.json();
        detail = body.detail || "";
    } catch {
        // Body was empty or not JSON - fall back to the status text.
    }

    const error = new Error(detail || `Request failed (${response.status}).`);
    error.status = response.status;

    return error;
}

// Send POST request with a JSON body
export async function postJson(endpoint, body) {

    const response = await fetch(
        API_BASE_URL + endpoint,
        {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(body), // converts a JavaScript object into a valid JSON
        }
    );

    if (!response.ok) {
        throw await toError(response);
    }

    return await response.json();
}

// Send DELETE request to backend
export async function deleteRequest(endpoint) {
    const response = await fetch(
        API_BASE_URL + endpoint,
        {
            method: "DELETE"
        }
    );

    if (!response.ok) {
        throw await toError(response);
    }

    // DELETE /sessions/{id} answers 204 No Content - there is no body to parse.
    if (response.status === 204) {
        return null;
    }

    return await response.json();
}

export { API_BASE_URL };
