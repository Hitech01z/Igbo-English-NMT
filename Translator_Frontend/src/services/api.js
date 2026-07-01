const API_BASE =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

const api = {
  async get(url) {
    const response = await fetch(`${API_BASE}${url}`);
    return response.json();
  },

  async post(url, data) {
    const response = await fetch(`${API_BASE}${url}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    return response.json();
  },
};

/* ===========================
   TRANSLATION
=========================== */

export const translate = (data) =>
  api.post("/translate/", data);

/* ===========================
   CONTRIBUTIONS
=========================== */

export const contribute = (data) =>
  api.post("/contribute/", data);

/* ===========================
   DATASET
=========================== */

export const getDataset = () =>
  api.get("/dataset/");

export const getDatasetProgress = () =>
  api.get("/dataset/progress");

/* ===========================
   DASHBOARD
=========================== */

export const getDashboardStats = () =>
  api.get("/dashboard/");

/* ===========================
   HISTORY
=========================== */

export const getHistory = () =>
  api.get("/history/");

/* ===========================
   DATASET EXPLORER
=========================== */

export const getDomains = () =>
  api.get("/dataset-explorer/domains");

export const getSentencesByDomain = (domain) =>
  api.get(`/dataset-explorer/${domain}`);

/* ===========================
   EXPORT DEFAULT
=========================== */

export default api;