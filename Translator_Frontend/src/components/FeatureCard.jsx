export default function FeatureCard({
  title,
  description,
}) {
  return (
    <div
      className="
        rounded-3xl
        border
        border-slate-800
        bg-slate-900
        p-5
        md:p-6
        transition
        hover:border-emerald-500
      "
    >
      <h3 className="text-lg md:text-xl font-semibold">
        {title}
      </h3>

      <p className="mt-4 text-sm md:text-base text-slate-400 leading-7">
        {description}
      </p>
    </div>
  );
}