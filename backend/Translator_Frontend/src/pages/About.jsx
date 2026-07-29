export default function About() {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 md:py-12">
      <h1 className="text-3xl md:text-4xl font-bold mb-8">
        About This Project
      </h1>

      <div className="rounded-3xl bg-slate-900 p-6 md:p-8">
        <div className="space-y-6 text-slate-300 leading-7 md:leading-8 text-sm md:text-base">
          <p>
            This project develops an English-Igbo Neural Machine
            Translation system.
          </p>

          <p>
            It supports dataset collection, cleaning, augmentation,
            evaluation, and deployment.
          </p>

          <p>
            Built as a Final Year Project using FastAPI, React,
            HuggingFace, and M2M100.
          </p>
        </div>
      </div>
    </div>
  );
}