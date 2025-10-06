import React from "react";
import { Link, useLocation } from "react-router-dom";
import { Search, Sun, Moon } from "lucide-react";

const navigationItems = [
  {
    title: "Classify",
    url: "/predict",
    icon: Search,
  },
];

export default function Layout({ children }) {
  const location = useLocation();
  const [theme, setTheme] = React.useState("dark");

  React.useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  return (
    <div className="min-h-screen flex w-full">
      <div className="w-64 border-r border-slate-700/50 dark:bg-slate-900/50 bg-white backdrop-blur-sm">
        <div className="border-b border-slate-700/50 dark:border-slate-700/50 border-slate-200 p-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Search className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="font-bold text-lg dark:text-white text-slate-900">CodeLock</h2>
              <p className="text-xs dark:text-slate-400 text-slate-600">AI Exoplanet Classifier</p>
            </div>
          </div>
        </div>
        
        <div className="p-3">
          <div className="space-y-1">
            {navigationItems.map((item) => (
              <Link
                key={item.title}
                to={item.url}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg mb-1 transition-all duration-200 ${
                  location.pathname === item.url
                    ? "bg-violet-500/20 dark:text-violet-400 text-violet-600"
                    : "dark:text-slate-300 text-slate-700 hover:bg-violet-500/10 dark:hover:text-violet-400 hover:text-violet-600"
                }`}
              >
                <item.icon className="w-4 h-4" />
                <span className="font-medium">{item.title}</span>
              </Link>
            ))}
          </div>
        </div>

        <div className="border-t border-slate-700/50 dark:border-slate-700/50 border-slate-200 p-4">
          <button
            onClick={toggleTheme}
            className="flex items-center gap-3 w-full px-3 py-2 rounded-lg hover:bg-slate-700/30 dark:hover:bg-slate-700/30 hover:bg-slate-200 transition-colors"
          >
            {theme === "dark" ? (
              <>
                <Sun className="w-4 h-4 dark:text-amber-400 text-amber-500" />
                <span className="text-sm font-medium dark:text-slate-300 text-slate-700">Light Mode</span>
              </>
            ) : (
              <>
                <Moon className="w-4 h-4 text-indigo-600" />
                <span className="text-sm font-medium text-slate-700">Dark Mode</span>
              </>
            )}
          </button>
        </div>
      </div>

      <main className="flex-1 flex flex-col">
        <header className="dark:bg-slate-900/30 bg-white/80 backdrop-blur-md border-b dark:border-slate-700/50 border-slate-200 px-6 py-4 md:hidden">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold dark:text-white text-slate-900">CodeLock</h1>
          </div>
        </header>

        <div className="flex-1 overflow-auto">
          {children}
        </div>
      </main>
    </div>
  );
}
