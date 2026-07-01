export default function Pagination({
  page,
  total,
  setPage,
}) {
  return (
    <div className="mt-8 flex justify-center gap-4">
      <button
        disabled={page === 1}
        onClick={() => setPage(page - 1)}
        className="rounded-xl bg-slate-800 px-4 py-2"
      >
        Previous
      </button>

      <span>
        {page} / {total}
      </span>

      <button
        disabled={page === total}
        onClick={() => setPage(page + 1)}
        className="rounded-xl bg-slate-800 px-4 py-2"
      >
        Next
      </button>
    </div>
  );
}