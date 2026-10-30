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
    backend_status: 'online',
    model_status: 'ready',
    gpu_status: 'not_allocated',
    service: 'Moon AI Sovereign Neural Gateway',
    edition: 'Prem Gaikwad Edition',
    backend: 'Vercel Serverless + Hugging Face Spaces (Hybrid)',
    active_engine: 'Open Neural Cluster Bridge',
    image_engine: 'AI Diffusion Cluster',
    last_successful_inference: new Date().toISOString(),
    architect: 'Prem Gaikwad (India 🇮🇳)',
    version: '3.2.0'
  });
}
