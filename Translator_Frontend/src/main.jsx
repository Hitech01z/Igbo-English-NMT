import React from "react";
import ReactDOM from "react-dom/client";

import "./index.css";

import App from "./App";

import { BrowserRouter } from "react-router-dom";
import { DatasetProvider } from "./context/DatasetContext";
import { TranslationProvider } from "./context/TranslationContext";

ReactDOM.createRoot(
  document.getElementById("root")
).render(
  <BrowserRouter>
    <DatasetProvider>
      <TranslationProvider>
        <App />
      </TranslationProvider>
    </DatasetProvider>
  </BrowserRouter>
);