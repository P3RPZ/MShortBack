import './App.css';
import Home from "./pages/Home";
import About from "./pages/About";
import How from "./pages/How";
import { BrowserRouter, Routes, Route } from "react-router-dom";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />}>
          <Route path="about" element={<About />} />
          <Route path="how" element={<How />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
