export default function TranslationHistory({
  rows,
}) {
  return (
    <div className="space-y-4">
      {rows.map((row, index) => (
        <div
          key={index}
          className="rounded-2xl bg-slate-900 p-5"
        >
          <p>{row.input}</p>

          <hr className="my-3 border-slate-700" />

          <p className="text-emerald-400">
            {row.output}
          </p>
        </div>
      ))}
    </div>
  );
}