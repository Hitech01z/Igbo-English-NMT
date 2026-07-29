export default function TranslationCard({
  original,
  translated
}) {
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div className="rounded-2xl bg-slate-900 p-6">
        <h3 className="mb-3 text-slate-400">
          Original
        </h3>

        <p>{original}</p>
      </div>

      <div className="rounded-2xl bg-emerald-900/20 p-6">
        <h3 className="mb-3 text-emerald-400">
          Translation
        </h3>

        <p>{translated}</p>
      </div>
    </div>
  );
}