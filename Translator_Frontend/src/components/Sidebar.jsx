import { NavLink } from "react-router-dom";

const links = [
  ["Dashboard", "/dashboard"],
  ["Dataset", "/dataset"],
  ["History", "/history"],
  ["Contribute", "/contribute"],
];

export default function Sidebar() {
  return (
    <aside className="hidden w-64 border-r border-slate-800 p-6 lg:block">
      <div className="space-y-4">
        {links.map(([name, path]) => (
          <NavLink
            key={path}
            to={path}
            className="block rounded-xl px-4 py-3 text-slate-300 hover:bg-slate-800"
          >
            {name}
          </NavLink>
        ))}
      </div>
    </aside>
  );
}