import { useState } from "react";
import { contribute } from "../services/ContributionService";

export default function ContributionForm() {
  const [igbo, setIgbo] = useState("");
  const [english, setEnglish] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();

    await contribute({
      igbo,
      english,
    });

    setIgbo("");
    setEnglish("");

    alert("Contribution saved.");
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-6 rounded-3xl bg-slate-900 p-5 md:p-8 shadow-lg"
    >
      <textarea
        value={igbo}
        onChange={(e) => setIgbo(e.target.value)}
        placeholder="Igbo text"
        className="w-full min-h-[150px] rounded-2xl border border-slate-700 bg-slate-800 p-4"
      />

      <textarea
        value={english}
        onChange={(e) => setEnglish(e.target.value)}
        placeholder="English text"
        className="w-full min-h-[150px] rounded-2xl border border-slate-700 bg-slate-800 p-4"
      />

      <button
        className="
          w-full
          sm:w-auto
          rounded-xl
          bg-emerald-600
          px-8
          py-3
          hover:bg-emerald-500
          transition
        "
      >
        Submit
      </button>
    </form>
  );
}