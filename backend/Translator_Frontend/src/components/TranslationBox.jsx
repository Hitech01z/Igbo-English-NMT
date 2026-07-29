export default function TranslationBox({
  value,
  onChange,
  placeholder,
  readOnly = false,
}) {
  return (
    <textarea
      value={value}
      readOnly={readOnly}
      onChange={(e) =>
        onChange?.(e.target.value)
      }
      placeholder={placeholder}
      rows={8}
      className="w-full rounded-3xl border border-slate-800 bg-slate-900 p-6 text-white outline-none"
    />
  );
}