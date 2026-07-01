import { useState } from "react";
import api from "../services/api";

export default function Translate() {
  const [sourceText, setSourceText] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [direction, setDirection] = useState("ig-en");
  const [loading, setLoading] = useState(false);

  const translate = async () => {
    if (!sourceText.trim()) return;

    setLoading(true);

    try {
     const result = await api.post(
  "/translate/",
  {
    text: sourceText,

    source_language:
      direction === "ig-en"
        ? "ig"
        : "en",

    target_language:
      direction === "ig-en"
        ? "en"
        : "ig"
  }
);

setTranslatedText(
  result.translation
);

      setTranslatedText(result.translation);
    } catch (error) {
      console.log(error);
      setTranslatedText("Translation failed.");
    } finally {
      setLoading(false);
    }
  };

  const switchDirection = () => {
    setDirection((prev) =>
      prev === "ig-en" ? "en-ig" : "ig-en"
    );

    setSourceText("");
    setTranslatedText("");
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between mb-8">
        <h1 className="text-3xl md:text-4xl font-bold">
          Translator
        </h1>

        <button
          onClick={switchDirection}
          className="w-full sm:w-auto px-5 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 transition"
        >
          {direction === "ig-en"
            ? "Igbo → English"
            : "English → Igbo"}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <textarea
          value={sourceText}
          onChange={(e) => setSourceText(e.target.value)}
          placeholder={
            direction === "ig-en"
              ? "Type Igbo text..."
              : "Type English text..."
          }
          className="w-full h-72 rounded-2xl border border-slate-700 bg-slate-900 p-4 resize-none"
        />

        <textarea
          value={translatedText}
          readOnly
          placeholder="Translation appears here..."
          className="w-full h-72 rounded-2xl border border-slate-700 bg-slate-800 p-4 resize-none"
        />
      </div>

      <button
        onClick={translate}
        disabled={loading}
        className="mt-6 w-full sm:w-auto px-8 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 transition"
      >
        {loading ? "Translating..." : "Translate"}
      </button>
    </div>
  );
}