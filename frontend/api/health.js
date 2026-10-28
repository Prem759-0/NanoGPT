// Vercel Serverless Function: Health endpoint
export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  return res.status(200).json({
    status: 'operational',
    service: 'Moon AI Sovereign Neural Gateway',
    edition: 'Prem Gaikwad Sovereign Edition',
    backend: 'Vercel Serverless + Hugging Face Spaces (Hybrid)',
    active_engine: 'Moon AI 24/7 Sovereign Neural Cloud',
    gpu_allocated: false,
    zero_downtime_active: true,
    architect: 'Prem Gaikwad (India 🇮🇳)',
    version: '3.1.0'
  });
}
