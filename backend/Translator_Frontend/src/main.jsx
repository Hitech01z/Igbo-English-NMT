import React from "react";
import ReactDOM from "react-dom/client";

import "./index.css";

import App from "./App";

import { BrowserRouter } from "react-router-dom";
import { TranslationProvider } from "./context/TranslationContext";

ReactDOM.createRoot(
  document.getElementById("root")
).render(
  <BrowserRouter>
    <TranslationProvider>
      <App />
    </TranslationProvider>
  </BrowserRouter>
);