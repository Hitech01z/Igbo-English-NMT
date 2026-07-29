export default function LanguageSwitcher({
  source,
  target,
  onSourceChange,
  onTargetChange,
  onSwap,
}) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-4">
      <select
        value={source}
        onChange={(e) =>
          onSourceChange(e.target.value)
        }
        className="rounded-xl bg-slate-800 px-4 py-3"
      >
        <option value="ig">Igbo</option>
        <option value="en">English</option>
      </select>

      <button
        onClick={onSwap}
        className="rounded-full bg-emerald-600 px-5 py-3"
      >
        ⇄
      </button>

      <select
        value={target}
        onChange={(e) =>
          onTargetChange(e.target.value)
        }
        className="rounded-xl bg-slate-800 px-4 py-3"
      >
        <option value="en">English</option>
        <option value="ig">Igbo</option>
      </select>
    </div>
  );
}