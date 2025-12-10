import { AttendanceRaw, ProcessedRecord, TaxonomyRaw } from '../types';

// Helper to parse dates (Excel serial or String)
const parseDate = (value: string | number | null | undefined): Date | null => {
  if (!value) return null;
  
  // Excel Serial Date
  if (typeof value === 'number') {
    const date = new Date(Math.round((value - 25569) * 86400 * 1000));
    return date;
  }
  
  // String Date
  const date = new Date(value);
  return isNaN(date.getTime()) ? null : date;
};

const normalizeString = (str: any): string => {
  return String(str || "").trim();
};

const parseBoolean = (val: any): boolean => {
  const s = String(val).toLowerCase();
  return s === "true" || s === "yes" || s === "1";
};

export const processFiles = async (
  attendanceFiles: File[],
  taxonomyFile: File | null
): Promise<ProcessedRecord[]> => {
  const taxonomyMap = new Map<string, string>();

  // 1. Process Taxonomy if exists (The Lookup Table)
  if (taxonomyFile) {
    const taxData = await parseFile<TaxonomyRaw>(taxonomyFile);
    taxData.forEach(row => {
      // Left Join Key: Career Development Workshop Titles
      const key = normalizeString(row["Career Development Workshop Titles"]).toLowerCase();
      if (key) {
        taxonomyMap.set(key, normalizeString(row["Sub-Category"]));
      }
    });
  }

  // 2. Process Attendance Files (The Fact Table)
  let allRecords: ProcessedRecord[] = [];

  for (const file of attendanceFiles) {
    const rawData = await parseFile<AttendanceRaw>(file);
    
    const processed = rawData.map(row => {
      // Join Logic
      const eventName = normalizeString(row["Event Name"]);
      const eventKey = eventName.toLowerCase();
      const subCategory = taxonomyMap.get(eventKey) || "Uncategorized";
      
      const attDate = parseDate(row["Attended Date"]);
      const regDate = parseDate(row["Registered Date"]);
      
      // Critical Step: Detect Year (2023, 2024, 2025)
      // Fallback to Registered Date if Attended Date is missing, else 0
      const year = attDate ? attDate.getFullYear() : (regDate ? regDate.getFullYear() : 0);

      // Extract Time Hour
      let timeStr = "00:00";
      if (row["Attended Time"]) {
        if (typeof row["Attended Time"] === 'number') {
           // Excel fraction of day
           const totalSeconds = Math.floor(row["Attended Time"] * 86400);
           const hours = Math.floor(totalSeconds / 3600);
           const minutes = Math.floor((totalSeconds % 3600) / 60);
           timeStr = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
        } else {
           timeStr = String(row["Attended Time"]);
        }
      }

      return {
        id: normalizeString(row.SIMID),
        eventName: eventName,
        status: normalizeString(row["Attendance Status"]) as any,
        attendedDate: attDate,
        attendedTimeStr: timeStr,
        registeredDate: regDate,
        university: normalizeString(row["University Program"]),
        citizenship: normalizeString(row["Citizenship"]),
        gradTerm: normalizeString(row["Expected Grad Term"]),
        isWalkIn: parseBoolean(row["Is Walk In"]),
        subCategory,
        year
      };
    });

    allRecords = [...allRecords, ...processed];
  }

  return allRecords;
};

const parseFile = <T>(file: File): Promise<T[]> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const data = e.target?.result;
      try {
        const workbook = window.XLSX.read(data, { type: 'binary' });
        const sheetName = workbook.SheetNames[0];
        const sheet = workbook.Sheets[sheetName];
        const json = window.XLSX.utils.sheet_to_json(sheet);
        resolve(json as T[]);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = (err) => reject(err);
    reader.readAsBinaryString(file);
  });
};