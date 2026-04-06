
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Home from './components/Home';
import PredictDashboard from './components/PredictDashboard';
import ForecastDashboard from './components/ForecastDashboard';
import ThemeToggle from './components/ThemeToggle';

export type ViewState = 'home' | 'prediction' | 'forecasting';

const App: React.FC = () => {
  const [view, setView] = useState<ViewState>('home');
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  return (
    <div className={`min-h-screen transition-colors duration-300 ${isDark ? 'bg-[#0F172A] text-slate-100' : 'bg-[#F4F7FB] text-slate-900'} overflow-x-hidden`}>
      {/* Global Header Elements */}
      <div className="fixed top-6 right-6 z-[60] flex items-center gap-4">
        <ThemeToggle isDark={isDark} toggle={() => setIsDark(!isDark)} />
      </div>

      <AnimatePresence mode="wait">
        {view === 'home' && (
          <motion.div
            key="home"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ x: -100, opacity: 0 }}
            transition={{ duration: 0.5, ease: "easeInOut" }}
          >
            <Home onNavigate={setView} isDark={isDark} />
          </motion.div>
        )}

        {view === 'prediction' && (
          <motion.div
            key="prediction"
            initial={{ x: 100, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 100, opacity: 0 }}
            transition={{ duration: 0.5, ease: "easeInOut" }}
          >
            <PredictDashboard onBack={() => setView('home')} isDark={isDark} />
          </motion.div>
        )}

        {view === 'forecasting' && (
          <motion.div
            key="forecasting"
            initial={{ x: 100, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 100, opacity: 0 }}
            transition={{ duration: 0.5, ease: "easeInOut" }}
          >
            <ForecastDashboard onBack={() => setView('home')} isDark={isDark} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;
