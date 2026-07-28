import { useState } from "react";
import { submitContribution } from "../services/ContributionService";

export default function Contribute() {

    const [igbo, setIgbo] = useState("");
    const [english, setEnglish] = useState("");
    const [message, setMessage] = useState("");

    const submit = async () => {

        if (!igbo || !english) return;

        const result = await submitContribution(
            igbo,
            english
        );

        setMessage(result.message);

        setIgbo("");
        setEnglish("");
    };

    return (

        <div className="max-w-4xl mx-auto p-8">

            <h1 className="text-4xl font-bold mb-8">

                Contribute Dataset

            </h1>

            <textarea
                value={igbo}
                onChange={(e)=>setIgbo(e.target.value)}
                placeholder="Igbo sentence"
                className="w-full h-32 p-4 rounded-xl bg-slate-900 border border-slate-700 mb-5"
            />

            <textarea
                value={english}
                onChange={(e)=>setEnglish(e.target.value)}
                placeholder="English sentence"
                className="w-full h-32 p-4 rounded-xl bg-slate-900 border border-slate-700"
            />

            <button
                onClick={submit}
                className="mt-6 px-8 py-3 rounded-xl bg-blue-600 hover:bg-blue-500"
            >
                Submit
            </button>

            <p className="mt-4 text-green-400">

                {message}

            </p>

        </div>

    );

}