import API_URL from "./api";

export async function translateText(text, direction) {
  const response = await fetch(
    `${API_URL}/translate`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        direction,
      }),
    }
  );

  return response.json();
}