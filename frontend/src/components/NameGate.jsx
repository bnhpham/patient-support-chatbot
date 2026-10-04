import { useState } from "react";

// Verification: the patient identifies themselves by typing their full name. 
// Intentionally simplified - a real system would need proper authentication (e.g. SMS/email OTP) here instead.
function NameGate({ onVerify, loading, error }) {

    const [name, setName] = useState("");

    function handleSubmit(event) {
        event.preventDefault();

        const trimmed = name.trim();

        if (!trimmed || loading) { return; }

        onVerify(trimmed);
    }

    return (
        <section className="name-gate">
            <div className="panel-header">VERIFY</div>

            <form className="name-gate__form" onSubmit={handleSubmit}>
                <label className="name-gate__label" htmlFor="full-name">
                    Enter your full name to continue
                </label>

                <input id="full-name" type="text" placeholder="e.g. James Whitfield" value={name}
                onChange={(event) => setName(event.target.value)} disabled={loading} autoFocus/>

                <button type="submit" className="retro-button" disabled={loading || !name.trim()}>
                    {loading ? "CHECKING..." : "START CHAT"}
                </button>

                {error && (<p className="error-message">{error}</p>)}
            </form>
        </section>
    );
}

export default NameGate;
