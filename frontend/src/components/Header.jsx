import { HandMetal } from "lucide-react";
import { useNavigate } from "react-router-dom";

// components/Header.jsx
const Header = () => {
  const navigate = useNavigate();

  return (
    <header className="w-full py-4 px-6  bg-gray-900  shadow-lg">
      <div className="container mx-auto max-w-6xl flex items-center justify-between">
        {/* Logo + Title */}
        <div
          className="flex items-center cursor-pointer"
          onClick={() => {
            navigate("/");
          }}
        >
          <div className="w-8 h-8 rounded-full bg-blue-500/0 flex items-center justify-center text-white mr-3 shadow-md">
            <HandMetal className="w-6 h-6 text-blue-500" />
          </div>
          <h1 className="text-xl md:text-2xl font-bold text-white tracking-wide">
            Sign<span className="text-blue-400">Speak</span>
          </h1>
        </div>

        {/* Right Side Icons */}
        <div className="flex items-center space-x-6">
          <a
            href="#"
            className="text-gray-300 hover:text-blue-400 transition-colors duration-300"
            aria-label="GitHub"
          >
            <i className="fab fa-github fa-lg"></i>
          </a>
          <a
            href="#"
            className="text-gray-300 hover:text-blue-400 transition-colors duration-300"
            aria-label="Help"
          >
            <i className="fas fa-question-circle fa-lg"></i>
          </a>
        </div>
      </div>
    </header>
  );
};

export default Header;
