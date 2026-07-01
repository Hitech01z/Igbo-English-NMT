import api from "./api";

export async function getDashboardStats() {
  return api.get("/dashboard/");
}