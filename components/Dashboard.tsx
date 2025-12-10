import React, { useState, useEffect } from 'react';
import { ProcessedRecord, NotebookCellResult } from '../types';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  AreaChart, Area, LineChart, Line, ComposedChart, Scatter, Cell 
} from 'recharts';
import { Heatmap } from './charts/Heatmap';
import { Play, RotateCcw, AlertTriangle } from 'lucide-react';

// Color Palette for Years
const COLORS = {
  2023: "#94a3b8", // Light Grey/Blue
  2024: "#3b82f6", // Medium Blue
  2025: "#1e3a8a", // Dark Blue (Highlight)
  default: "#6366f1"
};

// ----------------------------------------------------------------------
// DEFAULT ALGORITHMS (The "Python" Logic mapped to JS)
// ----------------------------------------------------------------------

const DEFAULT_CODE_Q1 = `
// Q1: Reach Efficiency (Single-Run vs Multi-Run)
// Goal: 100% Stacked Bar Chart

const results = [];
const years = [2023, 2024, 2025];

// Filter only Attended
const attended = data.filter(d => d.status === 'Attended');

years.forEach(year => {
  const yearData = attended.filter(d => d.year === year);
  
  // Group by Student ID
  const studentCounts = {};
  yearData.forEach(d => {
    if (d.id) studentCounts[d.id] = (studentCounts[d.id] || 0) + 1;
  });
  
  let single = 0;
  let multi = 0;
  Object.values(studentCounts).forEach(c => c === 1 ? single++ : multi++);
  
  const total = single + multi || 1;
  results.push({
    name: year.toString(),
    "Single-Run": parseFloat(((single/total)*100).toFixed(1)),
    "Multi-Run": parseFloat(((multi/total)*100).toFixed(1)),
    totalStudents: total
  });
});

return {
  stats: {
    "2023 Students": results.find(r => r.name === '2023')?.totalStudents || 0,
    "2025 Students": results.find(r => r.name === '2025')?.totalStudents || 0,
  },
  chartData: results,
  chartType: 'stacked-bar'
};
`;

const DEFAULT_CODE_Q2 = `
// Q2: Unique Participant Count
// Goal: Grouped Bar Chart comparing totals

const results = [];
const years = [2023, 2024, 2025];
const attended = data.filter(d => d.status === 'Attended');

years.forEach(year => {
  const yearData = attended.filter(d => d.year === year);
  const uniqueIDs = new Set(yearData.map(d => d.id).filter(Boolean));
  
  results.push({
    name: year.toString(),
    Count: uniqueIDs.size
  });
});

const count23 = results[0].Count;
const count25 = results[2].Count;
const growth = count23 > 0 ? ((count25 - count23)/count23 * 100).toFixed(1) + '%' : 'N/A';

return {
  stats: {
    "Growth (23-25)": growth,
    "Current Reach": count25
  },
  chartData: results,
  chartType: 'bar'
};
`;

const DEFAULT_CODE_Q3 = `
// Q3: Most Popular Days & Time Slots
// Goal: Faceted Heatmaps (3 subplots)

const years = [2023, 2024, 2025];
const attended = data.filter(d => d.status === 'Attended');
const heatmaps = {};

years.forEach(year => {
  const yearData = attended.filter(d => d.year === year);
  const counts = {};
  
  yearData.forEach(r => {
    if (r.attendedDate) {
      // Need real date object handling
      const dateObj = new Date(r.attendedDate);
      const day = dateObj.toLocaleDateString('en-US', { weekday: 'long' });
      const hour = parseInt(r.attendedTimeStr.split(':')[0], 10);
      const key = day + '-' + hour;
      counts[key] = (counts[key] || 0) + 1;
    }
  });

  heatmaps[year] = Object.entries(counts).map(([k, v]) => {
    const [day, h] = k.split('-');
    return { day, hour: parseInt(h), value: v };
  });
});

return {
  stats: { "Status": "Heatmaps Generated" },
  chartData: [],
  chartType: 'heatmap',
  extra: heatmaps
};
`;

const DEFAULT_CODE_Q4 = `
// Q4: Attendance by University
// Goal: Multi-Line or Grouped Bar for Top 5

const years = [2023, 2024, 2025];
const attended = data.filter(d => d.status === 'Attended');

// 1. Find Top 5 Universities overall
const allCounts = {};
attended.forEach(d => {
  const u = d.university || "Unknown";
  allCounts[u] = (allCounts[u] || 0) + 1;
});
const topUnis = Object.entries(allCounts)
  .sort((a,b) => b[1] - a[1])
  .slice(0, 5)
  .map(x => x[0]);

// 2. Build Data for Chart
const chartData = [];
years.forEach(year => {
  const yearData = attended.filter(d => d.year === year);
  const yearCounts = {};
  yearData.forEach(d => {
    const u = d.university || "Unknown";
    yearCounts[u] = (yearCounts[u] || 0) + 1;
  });

  const record = { name: year.toString() };
  topUnis.forEach(uni => {
    record[uni] = yearCounts[uni] || 0;
  });
  chartData.push(record);
});

return {
  stats: { "Top Partner": topUnis[0] },
  chartData: chartData,
  chartType: 'grouped-bar',
  extra: topUnis // Pass keys to render bars
};
`;

const DEFAULT_CODE_Q5 = `
// Q5: Workshop Attendance by Sub-Category
// Goal: Clustered Column Chart (Hue = Year)

const years = [2023, 2024, 2025];
const attended = data.filter(d => d.status === 'Attended');

// Get all categories
const categories = Array.from(new Set(attended.map(d => d.subCategory)));
const chartData = [];

categories.forEach(cat => {
  const record = { name: cat };
  years.forEach(year => {
    const count = attended.filter(d => d.year === year && d.subCategory === cat).length;
    record[year] = count;
  });
  chartData.push(record);
});

// Sort by 2025 volume
chartData.sort((a, b) => b[2025] - a[2025]);

return {
  stats: { "Top Topic '25": chartData[0]?.name },
  chartData: chartData.slice(0, 8), // Top 8
  chartType: 'grouped-bar-years' 
};
`;

const DEFAULT_CODE_Q6 = `
// Q6: Student Type (Local vs International)
// Goal: Stacked Area Chart

const years = [2023, 2024, 2025];
const attended = data.filter(d => d.status === 'Attended');
const chartData = [];

years.forEach(year => {
  const yearData = attended.filter(d => d.year === year);
  const unique = new Set();
  let local = 0, intl = 0;
  
  yearData.forEach(d => {
    if(!d.id || unique.has(d.id)) return;
    unique.add(d.id);
    
    const cit = (d.citizenship || "").toLowerCase();
    if(cit.includes('singapore') || cit.includes('pr')) local++;
    else intl++;
  });

  chartData.push({
    name: year.toString(),
    Local: local,
    International: intl
  });
});

return {
  stats: { "Analysis": "Diversity Split" },
  chartData: chartData,
  chartType: 'area'
};
`;

const DEFAULT_CODE_Q7 = `
// Q7: Attendance by Expected Graduation Period
// Goal: 100% Stacked Bar (Final Year vs Pre-Final)

const years = [2023, 2024, 2025];
const attended = data.filter(d => d.status === 'Attended');
const chartData = [];

years.forEach(year => {
  const yearData = attended.filter(d => d.year === year);
  let finalYear = 0;
  let preFinal = 0;
  
  yearData.forEach(d => {
    if (!d.gradTerm || !d.attendedDate) return;
    // Simple parsing logic: Extract year from "Spring 2025"
    const match = String(d.gradTerm).match(/\\d{4}/);
    if(match) {
      const gradYear = parseInt(match[0]);
      if (gradYear - year <= 1) finalYear++;
      else preFinal++;
    }
  });
  
  const total = finalYear + preFinal || 1;
  chartData.push({
    name: year.toString(),
    "Final Year": parseFloat(((finalYear/total)*100).toFixed(1)),
    "Pre-Final": parseFloat(((preFinal/total)*100).toFixed(1))
  });
});

return {
  stats: { "Focus": "Audience Maturity" },
  chartData: chartData,
  chartType: 'stacked-bar'
};
`;

const DEFAULT_CODE_Q8 = `
// Q8: Attribution Based on Registration Timing
// Goal: Grouped Bar Chart

const years = [2023, 2024, 2025];
const attended = data.filter(d => d.status === 'Attended');

// Bins
const bins = ["Same Day", "1-3 Days", ">1 Week"];
const chartData = bins.map(b => ({ name: b }));

years.forEach(year => {
  const yearData = attended.filter(d => d.year === year);
  let c1=0, c2=0, c3=0;
  
  yearData.forEach(d => {
    if (d.isWalkIn) { c1++; return; }
    if (d.attendedDate && d.registeredDate) {
      const diff = new Date(d.attendedDate) - new Date(d.registeredDate);
      const days = diff / (1000 * 3600 * 24);
      if (days <= 0.9) c1++;
      else if (days <= 3.9) c2++;
      else c3++;
    } else {
      c1++; // Default fallback
    }
  });

  chartData[0][year] = c1;
  chartData[1][year] = c2;
  chartData[2][year] = c3;
});

return {
  stats: { "Metric": "Planning Behavior" },
  chartData: chartData,
  chartType: 'grouped-bar-years'
};
`;

const DEFAULT_CODE_Q9 = `
// Q9: Walk-In Analysis
// Goal: Line Chart (Trend)

const years = [2023, 2024, 2025];
const attended = data.filter(d => d.status === 'Attended');
const chartData = [];

years.forEach(year => {
  const yearData = attended.filter(d => d.year === year);
  const total = yearData.length || 1;
  const walkIns = yearData.filter(d => d.isWalkIn).length;
  
  chartData.push({
    name: year.toString(),
    "Walk-In Rate": parseFloat(((walkIns/total)*100).toFixed(1))
  });
});

return {
  stats: { "Current Rate": chartData[2]["Walk-In Rate"] + "%" },
  chartData: chartData,
  chartType: 'line'
};
`;

const DEFAULT_CODE_Q10 = `
// Q10: No-Show Analysis
// Goal: Side-by-Side Bar (Comparison)

const years = [2023, 2024, 2025];
// Use ALL data, not just attended
const allData = data; 
const categories = Array.from(new Set(allData.map(d => d.subCategory)));
const chartData = [];

categories.forEach(cat => {
  const record = { name: cat };
  years.forEach(year => {
    const subset = allData.filter(d => d.year === year && d.subCategory === cat);
    if (subset.length > 5) { // Filter out tiny events
        const absent = subset.filter(d => d.status !== 'Attended').length;
        record[year] = parseFloat(((absent/subset.length)*100).toFixed(1));
    } else {
        record[year] = 0;
    }
  });
  chartData.push(record);
});

// Sort by 2025 no-show rate desc
chartData.sort((a,b) => b[2025] - a[2025]);

return {
  stats: { "Highest Dropout": chartData[0]?.name },
  chartData: chartData.slice(0, 10),
  chartType: 'grouped-bar-years'
};
`;


// ----------------------------------------------------------------------
// COMPONENT: Notebook Cell
// ----------------------------------------------------------------------

const NotebookCell: React.FC<{
  id: number;
  title: string;
  defaultCode: string;
  data: ProcessedRecord[];
}> = ({ id, title, defaultCode, data }) => {
  const [code, setCode] = useState(defaultCode);
  const [result, setResult] = useState<NotebookCellResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = () => {
    setError(null);
    try {
      // Create a Function from the string. 
      // Note: In a real "Sandbox" we might use a worker, but for this PRD client-side requirement, this is the way.
      const func = new Function('data', code);
      const res = func(data);
      setResult(res);
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 mb-8 overflow-hidden">
      {/* Header */}
      <div className="bg-gray-50 px-6 py-4 border-b border-gray-200 flex items-center justify-between">
        <h3 className="font-bold text-gray-800 flex items-center gap-2">
          <span className="bg-blue-100 text-blue-700 w-8 h-8 rounded-full flex items-center justify-center text-sm">{id}</span>
          {title}
        </h3>
        <div className="text-xs text-gray-400 font-mono">JavaScript Sandbox Mode</div>
      </div>

      {/* Editor Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 border-b border-gray-100">
        <div className="lg:col-span-12 p-4 bg-[#f8f9fa]">
           <label className="block text-xs font-bold text-gray-500 uppercase mb-2 tracking-wider">Analysis Logic (Editable)</label>
           <textarea 
             className="w-full h-48 font-mono text-sm bg-white border border-gray-300 rounded-lg p-4 focus:ring-2 focus:ring-blue-500 focus:outline-none text-gray-700"
             value={code}
             onChange={(e) => setCode(e.target.value)}
             spellCheck={false}
           />
           <div className="mt-3 flex justify-end">
              <button 
                onClick={runAnalysis}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-md font-medium text-sm transition-colors shadow-sm"
              >
                <Play className="w-4 h-4 fill-current" />
                Run Analysis
              </button>
           </div>
        </div>
      </div>

      {/* Output Area */}
      {error && (
        <div className="p-6 bg-red-50 text-red-700 flex items-start gap-3 border-l-4 border-red-500">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <div>
            <div className="font-bold">Logic Error</div>
            <div className="font-mono text-sm mt-1">{error}</div>
          </div>
        </div>
      )}

      {result && !error && (
        <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[300px]">
          {/* Stats Box (Left) */}
          <div className="lg:col-span-3 border-r border-gray-100 p-6 bg-gray-50/50">
            <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4">Key Statistics</h4>
            <div className="space-y-4">
              {Object.entries(result.stats).map(([key, val]) => (
                <div key={key} className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                  <div className="text-xs text-gray-500 mb-1">{key}</div>
                  <div className="text-2xl font-bold text-blue-900">{val}</div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Graph Box (Right) */}
          <div className="lg:col-span-9 p-6 relative">
            <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Visualization</h4>
            <div className="h-[300px] w-full">
              {/* Render appropriate chart based on result.chartType */}
              <ResponsiveContainer width="100%" height="100%">
                {renderChart(result)}
              </ResponsiveContainer>
            </div>
            {/* Custom Legend for Year Comparison */}
            {(result.chartType === 'grouped-bar-years' || result.chartType === 'area' || result.chartType === 'stacked-bar' || result.chartType === 'line') && (
               <div className="flex justify-center gap-6 mt-2">
                 {[2023, 2024, 2025].map(y => (
                   <div key={y} className="flex items-center gap-2 text-xs text-gray-600">
                      <span className="w-3 h-3 rounded-full" style={{backgroundColor: COLORS[y as keyof typeof COLORS]}}></span>
                      {y}
                   </div>
                 ))}
               </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Chart Factory
const renderChart = (res: NotebookCellResult) => {
  const { chartType, chartData, extra } = res;

  if (chartType === 'heatmap') {
    // Special Layout for 3 Heatmaps
    return (
       <div className="grid grid-cols-3 gap-2 h-full">
          {[2023, 2024, 2025].map(year => (
            <div key={year} className="h-full overflow-hidden">
               <Heatmap data={extra[year]} title={year.toString()} />
            </div>
          ))}
       </div>
    ) as any;
  }

  if (chartType === 'stacked-bar') {
    // Keys: Single-Run, Multi-Run OR Final Year, Pre-Final
    const keys = Object.keys(chartData[0]).filter(k => k !== 'name' && k !== 'totalStudents');
    return (
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip cursor={{fill: '#f3f4f6'}} />
        {keys.map((key, idx) => (
          <Bar key={key} dataKey={key} stackId="a" fill={idx === 0 ? COLORS[2023] : (idx === 1 ? COLORS[2024] : COLORS[2025])} />
        ))}
        <Legend />
      </BarChart>
    );
  }

  if (chartType === 'grouped-bar') {
    // For Q2 (Simple Count) or Q4 (Uni Top 5 - Multi Series?)
    // If simple count, just one bar. If multi keys, grouped.
    const keys = Object.keys(chartData[0]).filter(k => k !== 'name');
    if (keys.length === 1 && keys[0] === 'Count') {
       return (
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="Count" name="Unique Participants">
            {chartData.map((entry: any, index: number) => (
               <React.Fragment key={`cell-${index}`}>
                 {entry.name === '2023' && <Cell fill={COLORS[2023]} />}
                 {entry.name === '2024' && <Cell fill={COLORS[2024]} />}
                 {entry.name === '2025' && <Cell fill={COLORS[2025]} />}
               </React.Fragment>
            ))}
            {/* Fallback fill if cells don't work in this version */}
            <Cell fill={COLORS.default} /> 
          </Bar>
        </BarChart>
       );
    }
    // Grouped Bar for Q4 (Unis)
    return (
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        {extra.map((key: string, idx: number) => (
           <Bar key={key} dataKey={key} fill={['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe'][idx % 5]} />
        ))}
      </BarChart>
    );
  }

  if (chartType === 'grouped-bar-years') {
    // X-Axis = Category, Bars = Years
    return (
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" tick={{fontSize: 10}} interval={0} />
        <YAxis />
        <Tooltip />
        <Bar dataKey="2023" fill={COLORS[2023]} />
        <Bar dataKey="2024" fill={COLORS[2024]} />
        <Bar dataKey="2025" fill={COLORS[2025]} />
      </BarChart>
    );
  }

  if (chartType === 'area') {
    return (
      <AreaChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Area type="monotone" dataKey="Local" stackId="1" stroke={COLORS[2024]} fill={COLORS[2024]} fillOpacity={0.6} />
        <Area type="monotone" dataKey="International" stackId="1" stroke={COLORS[2023]} fill={COLORS[2023]} fillOpacity={0.6} />
      </AreaChart>
    );
  }

  if (chartType === 'line') {
    return (
       <LineChart data={chartData}>
         <CartesianGrid strokeDasharray="3 3" />
         <XAxis dataKey="name" />
         <YAxis />
         <Tooltip />
         <Line type="monotone" dataKey="Walk-In Rate" stroke={COLORS[2025]} strokeWidth={3} dot={{r: 5}} />
       </LineChart>
    );
  }

  return <div>Chart type not supported</div>;
};


// ----------------------------------------------------------------------
// MAIN EXPORT: Notebook Dashboard
// ----------------------------------------------------------------------

export const Dashboard: React.FC<{ data: ProcessedRecord[]; onBack: () => void }> = ({ data, onBack }) => {
  return (
    <div className="min-h-screen bg-gray-100 pb-20 font-sans">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-between mb-8">
          <div>
            <button onClick={onBack} className="flex items-center gap-1 text-sm font-medium text-gray-500 hover:text-gray-900 mb-2 transition-colors">
              <RotateCcw className="w-4 h-4" /> Reset & Upload New Data
            </button>
            <h1 className="text-2xl font-bold text-gray-900">Analysis Notebook</h1>
            <p className="text-gray-500 text-sm">Reviewing {data.length} records across 2023-2025</p>
          </div>
          <div className="flex items-center gap-2">
             <span className="px-3 py-1 bg-green-100 text-green-700 text-xs font-bold uppercase rounded-full">System Ready</span>
          </div>
        </div>

        <div className="space-y-6">
          <NotebookCell id={1} title="Overall Attendance Overview (Reach Efficiency)" data={data} defaultCode={DEFAULT_CODE_Q1} />
          <NotebookCell id={2} title="Unique Participant Count" data={data} defaultCode={DEFAULT_CODE_Q2} />
          <NotebookCell id={3} title="Most Popular Days & Time Slots" data={data} defaultCode={DEFAULT_CODE_Q3} />
          <NotebookCell id={4} title="Attendance by University" data={data} defaultCode={DEFAULT_CODE_Q4} />
          <NotebookCell id={5} title="Workshop Attendance by Sub-Category" data={data} defaultCode={DEFAULT_CODE_Q5} />
          <NotebookCell id={6} title="Student Type (Local vs. International)" data={data} defaultCode={DEFAULT_CODE_Q6} />
          <NotebookCell id={7} title="Attendance by Expected Graduation Period" data={data} defaultCode={DEFAULT_CODE_Q7} />
          <NotebookCell id={8} title="Attribution Based on Registration Timing" data={data} defaultCode={DEFAULT_CODE_Q8} />
          <NotebookCell id={9} title="Walk-In Analysis (Trend)" data={data} defaultCode={DEFAULT_CODE_Q9} />
          <NotebookCell id={10} title="No-Show Analysis" data={data} defaultCode={DEFAULT_CODE_Q10} />
        </div>
      </div>
    </div>
  );
};