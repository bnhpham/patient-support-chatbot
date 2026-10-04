import { NavLink } from "react-router-dom";

import "../styles/Navbar.css";

function Navbar() {
    return (
        <header className="navbar">
            <NavLink to="/" className="navbar__brand">Guardrails Challenge</NavLink>
        </header>
    );
}

export default Navbar;