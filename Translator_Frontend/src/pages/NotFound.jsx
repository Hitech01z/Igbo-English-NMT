export default function NotFound() {
  return (
    <div className="min-h-[70vh] flex items-center justify-center px-4">
      <div className="text-center">
        <h1 className="text-5xl md:text-7xl font-bold">
          404
        </h1>

        <p className="mt-4 text-slate-400 text-sm md:text-base">
          Page not found.
        </p>

        <Link
          to="/"
          className="
            inline-block
            mt-8
            rounded-xl
            bg-emerald-600
            px-6
            py-3
            hover:bg-emerald-500
            transition
          "
        >
          Go Home
        </Link>
      </div>
    </div>
  );
}