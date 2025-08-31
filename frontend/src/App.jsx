// App.jsx
import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./routes/HomePage";
import Header from "./components/Header";
import Footer from "./components/Footer";
import TranslationPage from "./routes/translationPage";

function App() {
  return (
    <BrowserRouter>
      <Header />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/translate" element={<TranslationPage />} />
      </Routes>
      <Footer/>
    </BrowserRouter>
  );
}

export default App;
