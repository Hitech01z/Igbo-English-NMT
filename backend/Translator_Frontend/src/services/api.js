import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export async function translate(text, direction) {
  const response = await api.post("/translate", {
    text,
    direction,
  });

  return response.data;
}

export async function getDataset() {
  const response = await api.get("/dataset");
  return response.data;
}

export async function getHealth() {
  const response = await api.get("/");
  return response.data;
}

export default api;