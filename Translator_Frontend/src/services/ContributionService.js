import api from "./api";

export async function contribute(data) {

  return api.post(
    "/contribute/",
    data
  );

}