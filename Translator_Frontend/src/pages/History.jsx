import { useEffect, useState } from "react";

import LoadingSpinner from "../components/LoadingSpinner";

import {
  getHistory
} from "../services/historyService";

export default function History() {

  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, []);

  async function loadHistory() {

    try {

      const data = await getHistory();

      setHistory(data);

    } catch (error) {

      console.log(error);

    } finally {

      setLoading(false);

    }

  }

  if (loading) {

    return <LoadingSpinner />;

  }

  if (history.length === 0) {

    return (

      <div className="max-w-7xl mx-auto px-4 py-12">

        <h1 className="text-3xl md:text-4xl font-bold mb-6">
          Translation History
        </h1>

        <p className="text-slate-400">
          No translations yet.
        </p>

      </div>

    );

  }

  return (

    <div className="max-w-7xl mx-auto px-4 py-8 md:py-12">

      <h1 className="text-3xl md:text-4xl font-bold mb-8 md:mb-10">
        Translation History
      </h1>

      <div className="space-y-4 md:space-y-6">

        {history.map((item, index) => (

          <div
            key={index}
            className="
              bg-slate-900
              p-4 md:p-6
              rounded-2xl md:rounded-3xl
              break-words
            "
          >

            <p className="text-emerald-400 text-sm md:text-base">
              {item.input}
            </p>

            <p className="mt-3 text-slate-300 text-sm md:text-base">
              {item.output}
            </p>

          </div>

        ))}

      </div>

    </div>

  );

}