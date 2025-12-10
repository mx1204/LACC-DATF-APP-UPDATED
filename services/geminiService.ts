import { GoogleGenAI } from "@google/genai";

export const generateInsights = async (metricsSummary: string): Promise<string> => {
  if (!process.env.API_KEY) {
    throw new Error("API Key is missing. Please check your environment configuration.");
  }

  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

  const prompt = `
    You are a senior data analyst for a University Career Center. 
    Analyze the following JSON summary of student attendance and workshop metrics. 
    Provide 3 high-impact strategic insights or recommendations (bullet points) regarding student engagement, operational timing, and workshop topics.
    Keep it concise and professional.
    
    Data Summary:
    ${metricsSummary}
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        systemInstruction: "You are a helpful and precise data analyst.",
      }
    });

    return response.text || "No insights generated.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Unable to generate insights at this time. Please try again later.";
  }
};