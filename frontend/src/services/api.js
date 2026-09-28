const API_BASE_URL = 'http://127.0.0.1:8000'

export async function submitRiskAssessment(payload) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(data.detail || 'Unable to fetch OA risk assessment right now.')
  }

  return data
}
