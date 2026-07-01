export default function Hero() {
  return (
    <section className="px-4 sm:px-6 py-16 md:py-24 text-center">
      <h1
        className="
          text-3xl
          sm:text-4xl
          md:text-5xl
          lg:text-6xl
          font-bold
          leading-tight
        "
      >
        Igbo ↔ English Neural Machine Translation
      </h1>

      <p
        className="
          mx-auto
          mt-6
          max-w-3xl
          text-base
          md:text-lg
          text-slate-400
          leading-7
        "
      >
        Building educational datasets and intelligent
        translation systems for preserving the Igbo language.
      </p>
    </section>
  );
}