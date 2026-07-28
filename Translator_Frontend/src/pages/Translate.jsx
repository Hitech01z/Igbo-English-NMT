import { useState } from "react";
import api from "../services/api";

export default function Translate() {
  const [sourceText, setSourceText] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [direction, setDirection] = useState("ig-en");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState(() => {

    const saved = localStorage.getItem("translation-history");

    return saved ? JSON.parse(saved) : [];

});

  const translate = async () => {
    if (!sourceText.trim()) return;

    setLoading(true);

    try {
      const response = await api.post("/translate", {
        text: sourceText,
        direction: direction,
      });

      setTranslatedText(response.data.translation);

      const newHistory = [

    {
        source: sourceText,
        translated: response.data.translation,
        direction,
        time: new Date().toLocaleTimeString(),
    },

    ...history,

];

setHistory(newHistory);

localStorage.setItem(
    "translation-history",
    JSON.stringify(newHistory)
);
    } catch (err) {
      console.error(err);
      setTranslatedText("Translation failed.");
    } finally {
      setLoading(false);
    }
  };

  const switchDirection = () => {
    setDirection((prev) => (prev === "ig-en" ? "en-ig" : "ig-en"));
    setSourceText("");
    setTranslatedText("");
  };

  const clearAll = () => {
    setSourceText("");
    setTranslatedText("");
  };

  const copyTranslation = async () => {
    if (!translatedText) return;

    await navigator.clipboard.writeText(translatedText);
    alert("Translation copied!");
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12">

      {/* Header */}

      <div className="flex flex-col sm:flex-row justify-between items-center mb-8 gap-4">
        <h1 className="text-3xl md:text-4xl font-bold">
          Translator
        </h1>

        <button
          onClick={switchDirection}
          className="px-5 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 transition"
        >
          {direction === "ig-en"
            ? "Igbo → English"
            : "English → Igbo"}
        </button>
      </div>

      {/* Translation Boxes */}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Source */}

        <div>

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

          <div className="text-sm text-slate-400 mt-2">
            {sourceText.length} characters
          </div>

        </div>

        {/* Translation */}

        <div>

          <textarea
            value={translatedText}
            readOnly
            placeholder="Translation appears here..."
            className="w-full h-72 rounded-2xl border border-slate-700 bg-slate-800 p-4 resize-none"
          />

          <div className="flex gap-3 mt-4">

            <button
              onClick={copyTranslation}
              className="px-4 py-2 rounded-lg bg-green-600 hover:bg-green-500"
            >
              📋 Copy
            </button>

            <button
              onClick={clearAll}
              className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500"
            >
              🗑 Clear
            </button>

          </div>

        </div>

      </div>

      {/* Translate Button */}

      <button
        onClick={translate}
        disabled={loading}
        className="mt-8 px-8 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 transition"
      >
        {loading ? "⏳ Translating..." : "🚀 Translate"}
      </button>

      {/* History */}

      <div className="mt-12">

        <h2 className="text-2xl font-bold mb-4">
          Translation History
        </h2>

        <div className="space-y-4">

          {history.length === 0 && (
            <p className="text-slate-400">
              No translations yet.
            </p>
          )}

          {history.map((item, index) => (

            <div
              key={index}
              className="bg-slate-900 border border-slate-700 rounded-xl p-4"
            >
              <div className="text-xs text-slate-400">
                {item.direction} • {item.time}
              </div>

              <div className="mt-2 font-semibold">
                {item.source}
              </div>

              <div className="mt-1 text-blue-400">
                {item.translated}
              </div>

            </div>

          ))}

        </div>

      </div>

    </div>
  );
}