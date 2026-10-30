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
      id: 'moon-sovereign-neural',
      name: '🌙 Moon AI Sovereign (Open Neural Cluster — By Prem Gaikwad)',
      type: 'cloud_neural',
      description: 'Conversational reasoning, Python coding, and multi-turn sovereign intelligence.',
      active: true,
      parameters: '7.6B Base / Open Neural Bridge',
      available: true
    },
    {
      id: 'moon-image-studio',
      name: '🎨 Moon AI Image Studio (AI Diffusion)',
      type: 'cloud_multimodal',
      description: 'Real AI text-to-image synthesis via GPU diffusion cluster.',
      active: true,
      parameters: 'FLUX / SD Diffusion',
      available: true
    },
    {
      id: 'moon-zerogpu-qwen',
      name: '💻 Moon AI ZeroGPU (Hugging Face Spaces)',
      type: 'cloud_zerogpu',
      description: 'Hugging Face Space ZeroGPU Qwen2.5-7B-Instruct engine.',
      active: false,
      parameters: '7.61B',
      available: true
    }
  ];

  return res.status(200).json(models);
}
