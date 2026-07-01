import { useEffect, useState } from "react";

import StatCard from "../components/StatCard";
import DatasetGrowthChart from "../components/DatasetGrowthChart";
import DomainChart from "../components/DomainChart";
import LoadingSpinner from "../components/LoadingSpinner";

import { getDashboardStats } from "../services/DashboardService";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  async function loadStats() {
    try {
      const data = await getDashboardStats();

      console.log("Dashboard data:", data);

      setStats(data);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return <LoadingSpinner />;
  }

  if (!stats) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-12">
        <h1 className="text-4xl font-bold mb-6">
          Dashboard
        </h1>

        <p className="text-slate-400">
          Failed to load dashboard data.
        </p>
      </div>
    );
  }

  const goal = 5000;

  const progress =
    (stats.total_sentences / goal) * 100;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 md:py-12">

      <h1 className="text-3xl md:text-4xl font-bold mb-8 md:mb-10">
        Dashboard
      </h1>

      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">

        <StatCard
          title="Total Sentences"
          value={stats.total_sentences}
        />

        <StatCard
          title="Education"
          value={stats.education}
        />

        <StatCard
          title="Translations"
          value={stats.translations}
        />

        <StatCard
          title="BLEU Score"
          value={stats.bleu_score}
        />

      </div>

      <div className="grid gap-8 mt-10 lg:grid-cols-2">

        <DatasetGrowthChart />

        <DomainChart />

      </div>

      <div className="mt-10 rounded-3xl bg-slate-900 p-5 md:p-6">

        <div className="flex flex-col gap-2 sm:flex-row sm:justify-between">

          <span>Dataset Progress</span>

          <span className="font-semibold">
            {stats.total_sentences}/{goal}
          </span>

        </div>

        <div className="mt-4 h-4 rounded-full bg-slate-800 overflow-hidden">

          <div
            className="h-full rounded-full bg-emerald-500 transition-all"
            style={{
              width: `${Math.min(progress, 100)}%`,
            }}
          />

        </div>

      </div>

    </div>
  );
}