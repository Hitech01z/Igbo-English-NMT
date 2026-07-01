import api from "./api";

export async function getDataset() {
  return api.get("/dataset/");
}