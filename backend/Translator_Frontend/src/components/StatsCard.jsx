export default function StatsCard({
  title,
  value
}) {
  return (
    <div className="rounded-3xl border border-slate-800 bg-slate-900 p-6">
      <h3 className="text-sm uppercase tracking-wider text-slate-400">
        {title}
      </h3>

      <p className="mt-4 text-3xl font-bold">
        {value}
      </p>
    </div>
  );
}