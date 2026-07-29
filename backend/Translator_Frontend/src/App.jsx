import { Routes, Route } from "react-router-dom";

import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

import Home from "./pages/Home";
import Translate from "./pages/Translate";
import Dashboard from "./pages/Dashboard";
import DatasetExplorer from "./pages/DatasetExplorer";
import Contribute from "./pages/Contribute";
import History from "./pages/History";
import About from "./pages/About";
import NotFound from "./pages/NotFound";

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col">
      <Navbar />

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/translate" element={<Translate />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/dataset" element={<DatasetExplorer />} />
          <Route path="/contribute" element={<Contribute />} />
          <Route path="/history" element={<History />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>

      <Footer />
    </div>
  );
}