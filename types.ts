export interface AttendanceRaw {
  SIMID: string | number;
  "Event Name": string;
  "Attendance Status": string;
  "Attended Date": string | number; // Can be Excel serial or string
  "Attended Time": string | number;
  "Registered Date": string | number;
  "University Program": string;
  "Citizenship": string;
  "Expected Grad Term": string | number;
  "Is Walk In": string | boolean;
}

export interface TaxonomyRaw {
  "Career Development Workshop Titles": string;
  "Sub-Category": string;
}

export interface ProcessedRecord {
  id: string; // SIMID
  eventName: string;
  status: "Attended" | "Registered" | "Absent" | "Other";
  attendedDate: Date | null;
  attendedTimeStr: string; // HH:00 format
  registeredDate: Date | null;
  university: string;
  citizenship: string;
  gradTerm: string;
  isWalkIn: boolean;
  subCategory: string; // From Taxonomy
  year: number;
}

export enum AnalyticsView {
  UPLOAD = 'UPLOAD',
  NOTEBOOK = 'NOTEBOOK'
}

// For Chart Data
export interface ChartData {
  name: string;
  [key: string]: string | number;
}

export interface NotebookCellResult {
  stats: Record<string, string | number>;
  chartData: any[];
  chartType?: 'bar' | 'line' | 'area' | 'heatmap' | 'lollipop' | 'stacked-bar' | 'grouped-bar' | 'grouped-bar-years';
  extra?: any; // For heatmap data structure
}

// Global declaration for SheetJS
declare global {
  interface Window {
    XLSX: any;
  }
}