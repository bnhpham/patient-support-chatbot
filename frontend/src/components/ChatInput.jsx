function ChatInput({ value, onChange, onSubmit, disabled }) {

    // User has to press "enter" to submit
    function handleKeyDown(event) {
        if(event.key === "Enter") {
            // Stop the default browser action
            event.preventDefault();

            // Check whether the input field isn't currently disabled + Remove surrounding whitespace and check if there is actual text
            if(!disabled && value.trim()) {
                onSubmit();
            }
        }
    }

    // Text Input
    return (
        <div className="chat-input">
            {/*Capture user's text input*/}
            <input type="text" placeholder="Type your message..." value={value} onChange={(event) => onChange(event.target.value)}
            onKeyDown={handleKeyDown} disabled={disabled}/>
            
            {/*<button className="chat-input__send">SEND</button>*/}
        </div>
    )
}

export default ChatInput