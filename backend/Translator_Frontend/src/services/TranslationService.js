// src/services/translationService.js

import api from "./api";

export async function translateText(
  text,
  source_language,
  target_language
) {
  return api.post("/translate/", {
    text,
    source_language,
    target_language,
  });
}