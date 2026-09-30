const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

function getErrorMessage(detail, fallback) {
  if (typeof detail === 'string') return detail

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item?.msg)
      .filter(Boolean)

    if (messages.length > 0) return messages.join(' ')
  }

  return fallback
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options)

  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(getErrorMessage(data.detail, 'The AI screening service is unavailable right now.'))
  }

  return data
}

export function fetchModelInfo() {
  return requestJson('/model-info')
}

export function fetchEvaluation() {
  return requestJson('/evaluation')
}

export function buildPredictionPayload({ age, gender, BMI, VAS_score }) {
  return { age, gender, BMI, VAS_score }
}

export function submitRiskAssessment(payload) {
  return requestJson('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })
}
