export default function ConfirmModal({
  open,
  title,
  message,
  onConfirm,
  onCancel,
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black/70">
      <div className="w-full max-w-md rounded-3xl bg-slate-900 p-8">
        <h2 className="text-2xl font-bold">
          {title}
        </h2>

        <p className="mt-4 text-slate-400">
          {message}
        </p>

        <div className="mt-8 flex gap-4">
          <button
            onClick={onConfirm}
            className="flex-1 rounded-xl bg-emerald-600 py-3"
          >
            Confirm
          </button>

          <button
            onClick={onCancel}
            className="flex-1 rounded-xl bg-slate-700 py-3"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}