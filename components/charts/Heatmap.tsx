import React from 'react';

interface HeatmapProps {
  data: { day: string; hour: number; value: number }[];
  title?: string;
}

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const HOURS = Array.from({ length: 14 }, (_, i) => i + 8); // 8 AM to 9 PM

export const Heatmap: React.FC<HeatmapProps> = ({ data, title }) => {
  // Find max value for color scaling
  const maxVal = Math.max(...data.map(d => d.value), 1);

  const getValue = (day: string, hour: number) => {
    return data.find(d => d.day === day && d.hour === hour)?.value || 0;
  };

  const getColor = (val: number) => {
    if (val === 0) return 'bg-gray-50';
    const intensity = val / maxVal;
    if (intensity < 0.2) return 'bg-blue-100';
    if (intensity < 0.4) return 'bg-blue-300';
    if (intensity < 0.6) return 'bg-blue-500';
    if (intensity < 0.8) return 'bg-blue-700';
    return 'bg-blue-900';
  };

  return (
    <div className="overflow-x-auto bg-white p-4 rounded border border-gray-100">
      {title && <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3 text-center">{title}</h4>}
      <div className="min-w-[300px]">
        <div className="grid grid-cols-[60px_repeat(14,1fr)] gap-0.5 mb-1">
          <div className="text-xs font-semibold text-gray-400"></div>
          {HOURS.filter((_, i) => i % 2 === 0).map(h => (
            <div key={h} className="col-span-2 text-[10px] text-center text-gray-400 font-medium">
              {h}
            </div>
          ))}
        </div>
        {DAYS.map(day => (
          <div key={day} className="grid grid-cols-[60px_repeat(14,1fr)] gap-0.5 mb-0.5">
            <div className="text-[10px] font-medium text-gray-500 flex items-center">{day.substring(0, 3)}</div>
            {HOURS.map(hour => {
              const val = getValue(day, hour);
              return (
                <div
                  key={`${day}-${hour}`}
                  title={`${day} ${hour}:00 - ${val} students`}
                  className={`h-6 rounded-[1px] ${getColor(val)} hover:opacity-80 transition-opacity`}
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};