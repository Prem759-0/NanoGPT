// Vercel Serverless Function: Models endpoint
export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const models = [
    {
      id: 'moon-gemini-pro',
      name: '🌙 Moon AI Sovereign (Google Gemini Pro Engine — By Prem Gaikwad)',
      type: 'cloud_frontier',
      description: 'Ultra-fast frontier intelligence with 1M context window and deep coding capability.',
      active: true,
      parameters: 'Frontier Scale',
      available: true
    },
    {
      id: 'moon-groq-llama',
      name: '⚡ Moon AI Turbo (Llama 3.3 70B via Groq)',
      type: 'cloud_turbo',
      description: 'Sub-second conversational and coding engine running on LPUs.',
      active: false,
      parameters: '70B',
      available: true
    },
    {
      id: 'moon-zerogpu-qwen',
      name: '💻 Moon AI Open (ZeroGPU Cloud)',
      type: 'cloud_open',
      description: 'Hugging Face Space ZeroGPU fallback engine.',
      active: false,
      parameters: '7.61B',
      available: true
    }
  ];

  return res.status(200).json(models);
}
