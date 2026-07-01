import api from "./api";

export async function getHistory() {
  return api.get("/history/");
}