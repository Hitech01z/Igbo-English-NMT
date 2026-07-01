const domains = [
  "",
  "education",
  "health",
  "technology",
  "agriculture",
  "government",
  "culture",
  "general"
];

export default function FilterBar({
  value,
  onChange
}) {
  return (
    <select
      value={value}
      onChange={(e) =>
        onChange(e.target.value)
      }
      className="
        w-full
        rounded-2xl
        bg-slate-900
        border
        border-slate-700
        p-4
      "
    >
      {domains.map((domain) => (
        <option
          key={domain}
          value={domain}
        >
          {domain || "All Domains"}
        </option>
      ))}
    </select>
  );
}