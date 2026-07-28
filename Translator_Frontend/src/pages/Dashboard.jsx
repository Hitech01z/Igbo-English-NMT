import { useEffect, useState } from "react";
import { getDashboard } from "../services/DashboardService";

export default function Dashboard() {

    const [stats, setStats] = useState(null);

    useEffect(() => {

        getDashboard().then(setStats);

    }, []);

    if (!stats)
        return <p className="p-10">Loading...</p>;

    return (

        <div className="max-w-7xl mx-auto p-8">

            <h1 className="text-4xl font-bold mb-8">

                Dashboard

            </h1>

            <div className="grid md:grid-cols-4 gap-6">

                <Card
                    title="BLEU"
                    value={stats.bleu}
                />

                <Card
                    title="chrF++"
                    value={stats.chrf}
                />

                <Card
                    title="Dataset"
                    value={stats.dataset_size}
                />

                <Card
                    title="Model"
                    value="Transformer"
                />

                <Card
                    title="Training"
                    value={stats.training}
                />

                <Card
                    title="Validation"
                    value={stats.validation}
                />

                <Card
                    title="Test"
                    value={stats.test}
                />

                <Card
                    title="Vocabulary"
                    value={`${stats.igbo_vocab}/${stats.english_vocab}`}
                />

            </div>

        </div>

    );

}

function Card({ title, value }) {

    return (

        <div className="bg-slate-900 rounded-xl p-6 border border-slate-700">

            <p className="text-slate-400">

                {title}

            </p>

            <h2 className="text-3xl font-bold mt-2">

                {value}

            </h2>

        </div>

    );

}