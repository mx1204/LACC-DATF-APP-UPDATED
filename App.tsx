import React, { useState } from 'react';
import { UploadSection } from './components/FileUpload';
import { Dashboard } from './components/Dashboard';
import { processFiles } from './services/dataProcessor';
import { ProcessedRecord, AnalyticsView } from './types';

function App() {
  const [view, setView] = useState<AnalyticsView>(AnalyticsView.UPLOAD);
  const [data, setData] = useState<ProcessedRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleProcess = async (attFiles: File[], taxFile: File | null) => {
    setLoading(true);
    setError(null);
    try {
      const results = await processFiles(attFiles, taxFile);
      if (results.length === 0) {
        throw new Error("No valid records found. Please check file formatting.");
      }
      setData(results);
      setView(AnalyticsView.NOTEBOOK);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to process files. Ensure columns match requirements.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
      {view === AnalyticsView.UPLOAD && (
        <>
          <UploadSection onProcess={handleProcess} loading={loading} />
          {error && (
            <div className="max-w-4xl mx-auto mt-6 p-4 bg-red-50 text-red-700 border border-red-200 rounded-lg text-sm text-center flex items-center justify-center gap-2">
              <span className="font-bold">Error:</span> {error}
            </div>
          )}
          
          <div className="max-w-6xl mx-auto mt-12 grid grid-cols-1 md:grid-cols-3 gap-8 px-8">
             <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-100">
               <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center text-blue-600 font-bold mb-4">1</div>
               <h4 className="font-bold text-gray-900 mb-2">Upload Data</h4>
               <p className="text-sm text-gray-500">Drag and drop your raw attendance logs. The system stacks 2023, 2024, and 2025 files automatically.</p>
             </div>
             <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-100">
               <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center text-indigo-600 font-bold mb-4">2</div>
               <h4 className="font-bold text-gray-900 mb-2">Adjust Logic</h4>
               <p className="text-sm text-gray-500">Enter the Notebook view where calculation logic is exposed. Edit the code to tweak definitions on the fly.</p>
             </div>
             <div className="p-6 bg-white rounded-xl shadow-sm border border-gray-100">
               <div className="w-10 h-10 bg-emerald-100 rounded-lg flex items-center justify-center text-emerald-600 font-bold mb-4">3</div>
               <h4 className="font-bold text-gray-900 mb-2">Compare Years</h4>
               <p className="text-sm text-gray-500">Visualize side-by-side performance for 2023, 2024, and 2025 to identify long-term trends.</p>
             </div>
          </div>
        </>
      )}

      {view === AnalyticsView.NOTEBOOK && (
        <Dashboard 
          data={data} 
          onBack={() => setView(AnalyticsView.UPLOAD)} 
        />
      )}
    </div>
  );
}

export default App;