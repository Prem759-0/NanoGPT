// Vercel Serverless Function: Health endpoint
export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  return res.status(200).json({
    status: 'ok',
    edition: 'Prem Gaikwad Sovereign Edition',
    engine: 'Vercel Serverless + Gemini Pro / Groq Hybrid',
    uptime: '100% (Serverless)',
    version: '3.0.0'
  });
}
