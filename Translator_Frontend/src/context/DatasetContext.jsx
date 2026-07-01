import {
  createContext,
  useContext,
  useEffect,
  useState
} from "react";

import { getDataset } from "../services/api";

const DatasetContext = createContext();

export function DatasetProvider({ children }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDataset()
      .then(setRows)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <DatasetContext.Provider
      value={{
        rows,
        loading
      }}
    >
      {children}
    </DatasetContext.Provider>
  );
}

export function useDataset() {
  return useContext(DatasetContext);
}