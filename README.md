# Career Workshop Analytics Sandbox

A React + TypeScript Vite app for interactive attendance analytics and notebook-style exploration of student engagement across 2023, 2024, and 2025.

This repository is built to showcase a clean analytics workflow with:

- drag-and-drop attendance and taxonomy file upload
- flexible data normalization for Excel/CSV inputs
- an editable JavaScript sandbox for custom analysis logic
- rich visualizations using `recharts`
- year-over-year benchmark comparisons and trend analysis

## Key Features

- **Interactive Upload Dashboard**: Upload multiple attendance files and a taxonomy mapping file.
- **Data Normalization Engine**: Parses Excel serial dates, string dates, attendance status, and extracts workshop metadata.
- **Notebook-Style Analysis**: Ten editable analysis cells with default logic for attendance efficiency, participant growth, heatmaps, university comparisons, walk-in behavior, and no-show analysis.
- **Visualization Suite**: Bar charts, area charts, line charts, grouped comparisons, and heatmaps powered by `recharts`.
- **Hackathon-ready Portfolio Piece**: Demonstrates full-stack front-end analytics, data engineering, and extendable notebook-style UX.

## Tech Stack

- React 19
- TypeScript
- Vite
- Recharts
- Lucide icons
- Google Gemini GenAI client (optional) for future insights generation

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd "LACC-DATF-APP-UPDATED"
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open the local URL shown in the terminal.

## Usage

1. Upload attendance files for 2023, 2024, and 2025.
2. Optionally upload a taxonomy reference file to map workshop titles to sub-categories.
3. Click **Launch Interactive Notebook**.
4. Edit any notebook cell logic and rerun analysis to validate assumptions or explore alternative metrics.

## Project Structure

- `App.tsx` — main application state and view switching between upload and analysis notebook.
- `components/FileUpload.tsx` — file upload UI with drag-and-drop-style input and file preview.
- `components/Dashboard.tsx` — notebook dashboard with editable code cells and visualization rendering.
- `services/dataProcessor.ts` — attendance and taxonomy parsing, normalization, and record transformation.
- `services/geminiService.ts` — optional AI insights generator using Gemini.
- `types.ts` — shared TypeScript interfaces and analytics view definitions.

## Notes

- The upload pipeline supports both `.csv` and Excel files (.xlsx / .xls).
- The notebook runs analysis logic in-browser using editable JavaScript snippets.
- `services/geminiService.ts` is included for future AI insight integration but is not required to run the core app.

## Future Improvements

- add a secure backend to execute notebook code safely
- support multi-sheet workbooks and expanded data validation
- integrate analytic summaries from AI-generated insights
- add export/download for analysis reports and charts

## Run Scripts

- `npm run dev` — start the app locally
- `npm run build` — build production assets
- `npm run preview` — preview the production build
