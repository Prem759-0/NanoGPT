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
    engine: 'Moon AI 24/7 Sovereign Neural Cloud',
    device: 'Moon AI 24/7 Cloud Cluster',
    vram_gb: 16.0,
    uptime: '99.99% Continuous Zero-Downtime',
    architect: 'Prem Gaikwad (India 🇮🇳)',
    version: '3.0.0'
  });
}
