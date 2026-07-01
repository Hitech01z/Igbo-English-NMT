import { useState } from "react";
import { Link, NavLink } from "react-router-dom";

export default function Navbar() {
  const [open, setOpen] = useState(false);

  const links = [
    ["Home", "/"],
    ["Translate", "/translate"],
    ["Dashboard", "/dashboard"],
    ["Dataset", "/dataset"],
    ["Contribute", "/contribute"],
    ["History", "/history"],
    ["About", "/about"],
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-950/95 backdrop-blur-lg border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 py-4">

        <div className="flex items-center justify-between">

          <Link
            to="/"
            className="text-2xl font-bold text-emerald-400"
          >
            IgboNMT
          </Link>

          {/* Desktop */}
          <nav className="hidden md:flex items-center gap-6">
            {links.map(([label, path]) => (
              <NavLink
                key={label}
                to={path}
                className={({ isActive }) =>
                  `transition ${
                    isActive
                      ? "text-emerald-400"
                      : "text-slate-300 hover:text-white"
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>

          {/* Mobile button */}
          <button
            onClick={() => setOpen(!open)}
            className="md:hidden text-white text-2xl"
          >
            {open ? "✕" : "☰"}
          </button>

        </div>

        {/* Mobile menu */}
        {open && (
          <nav className="md:hidden mt-4 flex flex-col gap-3 pb-4">
            {links.map(([label, path]) => (
              <NavLink
                key={label}
                to={path}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 transition ${
                    isActive
                      ? "bg-emerald-600 text-white"
                      : "text-slate-300 hover:bg-slate-800"
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
        )}

      </div>
    </header>
  );
}