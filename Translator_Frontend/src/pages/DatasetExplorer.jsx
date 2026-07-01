import { useEffect, useState } from "react";

import DatasetTable from "../components/DatasetTable";
import SearchBar from "../components/SearchBar";
import FilterBar from "../components/FilterBar";
import LoadingSpinner from "../components/LoadingSpinner";

import { getDataset } from "../services/datasetService";

export default function DatasetExplorer() {
  const [rows, setRows] = useState([]);
  const [search, setSearch] = useState("");
  const [domain, setDomain] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRows();
  }, []);

  async function loadRows() {
    try {
      const data = await getDataset();

      console.log("Dataset:", data);

      setRows(data);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return <LoadingSpinner />;
  }

  if (!Array.isArray(rows)) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-12">
        <h1 className="text-4xl font-bold mb-6">
          Dataset Explorer
        </h1>

        <p className="text-red-400">
          Dataset API returned invalid data.
        </p>
      </div>
    );
  }

  const filtered = rows.filter((row) => {
    const text =
      `${row.igbo} ${row.english}`.toLowerCase();

    return (
      text.includes(search.toLowerCase()) &&
      (!domain ||
        row.domain?.toLowerCase() === domain)
    );
  });

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 md:py-12">

      <h1 className="text-3xl md:text-4xl font-bold mb-8">
        Dataset Explorer
      </h1>

      <div className="grid gap-4 md:grid-cols-2 mb-8">

        <SearchBar
          value={search}
          onChange={setSearch}
        />

        <FilterBar
          value={domain}
          onChange={setDomain}
        />

      </div>

      <DatasetTable rows={filtered} />

    </div>
  );
}