import React from 'react';
import { UploadCloud, FileText, Database, Layers } from 'lucide-react';

interface FileUploadProps {
  label: string;
  subLabel?: string;
  accept: string;
  multiple?: boolean;
  onFilesSelected: (files: File[]) => void;
  icon: React.ElementType;
  zone: 'A' | 'B';
}

const FileUpload: React.FC<FileUploadProps> = ({ label, subLabel, accept, multiple, onFilesSelected, icon: Icon, zone }) => {
  const [fileNames, setFileNames] = React.useState<string[]>([]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const filesArray: File[] = Array.from(e.target.files);
      setFileNames(filesArray.map(f => f.name));
      onFilesSelected(filesArray);
    }
  };

  const borderColor = zone === 'A' ? 'border-blue-300 hover:border-blue-400' : 'border-indigo-300 hover:border-indigo-400';
  const bgColor = zone === 'A' ? 'bg-blue-50/50' : 'bg-indigo-50/50';

  return (
    <div className="w-full h-full">
      <div className={`relative border-2 border-dashed ${borderColor} rounded-xl p-8 hover:bg-white transition-all ${bgColor} h-full flex flex-col justify-center items-center`}>
        <input
          type="file"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
          accept={accept}
          multiple={multiple}
          onChange={handleChange}
        />
        <div className="text-center pointer-events-none">
          <div className={`mx-auto h-12 w-12 rounded-full flex items-center justify-center mb-4 ${zone === 'A' ? 'bg-blue-100 text-blue-600' : 'bg-indigo-100 text-indigo-600'}`}>
            <Icon className="h-6 w-6" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900">{label}</h3>
          {subLabel && <p className="text-sm text-gray-500 mt-1">{subLabel}</p>}
          <p className="mt-4 text-xs font-medium text-gray-400 uppercase tracking-wide">
            {fileNames.length > 0 ? (
              <span className="text-emerald-600">{fileNames.length} file(s) loaded</span>
            ) : (
              <span>Drop files here or click</span>
            )}
          </p>
        </div>
      </div>
      {fileNames.length > 0 && (
        <div className="mt-3 bg-white p-3 rounded border border-gray-100 shadow-sm">
           <ul className="text-xs text-gray-500 list-disc list-inside">
             {fileNames.slice(0, 3).map((name, idx) => <li key={idx} className="truncate">{name}</li>)}
             {fileNames.length > 3 && <li>...and {fileNames.length - 3} more</li>}
           </ul>
        </div>
      )}
    </div>
  );
};

export const UploadSection: React.FC<{
  onProcess: (att: File[], tax: File | null) => void;
  loading: boolean;
}> = ({ onProcess, loading }) => {
  const [attFiles, setAttFiles] = React.useState<File[]>([]);
  const [taxFile, setTaxFile] = React.useState<File | null>(null);

  return (
    <div className="max-w-6xl mx-auto mt-10 p-8 bg-white rounded-2xl shadow-xl border border-gray-100">
      <div className="text-center mb-10">
        <span className="inline-block py-1 px-3 rounded-full bg-blue-100 text-blue-800 text-xs font-bold tracking-wide uppercase mb-3">
          Sandbox v2.0
        </span>
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">Career Workshop Analytics Sandbox</h1>
        <p className="text-gray-500 mt-2 max-w-2xl mx-auto">
          Interactive notebook environment for analyzing student engagement across 2023, 2024, and 2025.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
        <div className="flex flex-col">
          <label className="block text-sm font-bold text-gray-700 mb-2 uppercase tracking-wide">
            Zone A: Attendance Data
          </label>
          <FileUpload 
            label="Upload Raw Logs"
            subLabel="Support for 2023, 2024, 2025 CSV/Excel files"
            accept=".csv, .xlsx, .xls" 
            multiple 
            onFilesSelected={setAttFiles}
            icon={Layers}
            zone="A"
          />
        </div>
        
        <div className="flex flex-col">
          <label className="block text-sm font-bold text-gray-700 mb-2 uppercase tracking-wide">
             Zone B: Taxonomy Reference
          </label>
          <FileUpload 
            label="Upload Taxonomy"
            subLabel="Workshop mapping file for categorisation"
            accept=".csv, .xlsx, .xls" 
            onFilesSelected={(files) => setTaxFile(files[0])}
            icon={Database}
            zone="B"
          />
        </div>
      </div>

      <div className="flex justify-center">
        <button
          onClick={() => onProcess(attFiles, taxFile)}
          disabled={attFiles.length === 0 || loading}
          className={`py-4 px-12 rounded-full text-white font-bold text-lg shadow-lg transition-all transform hover:-translate-y-0.5
            ${attFiles.length === 0 || loading 
              ? 'bg-gray-300 cursor-not-allowed shadow-none' 
              : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 hover:shadow-xl'}`}
        >
          {loading ? (
             <span className="flex items-center gap-2">
               <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                 <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                 <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
               </svg>
               Processing Data...
             </span>
          ) : "Launch Interactive Notebook"}
        </button>
      </div>
    </div>
  );
};