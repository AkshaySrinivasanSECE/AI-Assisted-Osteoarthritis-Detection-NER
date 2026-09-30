export const symptomFields = [
  { name: 'affectedKnee', label: 'Affected knee', options: ['Left', 'Right', 'Both', 'None'] },
  { name: 'morningStiffness', label: 'Morning stiffness', options: ['Yes', 'No'] },
  { name: 'swelling', label: 'Knee swelling', options: ['Yes', 'No'] },
  { name: 'walkingDifficulty', label: 'Difficulty walking', options: ['None', 'Mild', 'Moderate', 'Severe'] },
  { name: 'stairDifficulty', label: 'Difficulty climbing stairs', options: ['None', 'Mild', 'Moderate', 'Severe'] },
  { name: 'standingDifficulty', label: 'Difficulty standing from a chair', options: ['None', 'Mild', 'Moderate', 'Severe'] },
]

export const historyFields = [
  { name: 'previousInjury', label: 'Previous knee injury', options: ['Yes', 'No'] },
  { name: 'previousSurgery', label: 'Previous knee surgery', options: ['Yes', 'No'] },
  { name: 'familyHistory', label: 'Family history of arthritis', options: ['Yes', 'No'] },
  { name: 'activityLevel', label: 'Physical activity level', options: ['Low', 'Moderate', 'High'] },
  { name: 'occupationType', label: 'Occupation type', options: ['Mostly sitting', 'Moderate movement', 'Heavy physical work'] },
]

export function calculateBMI(heightCm, weightKg) {
  const heightMeters = Number(heightCm) / 100
  const weight = Number(weightKg)
  if (!heightMeters || !weight || heightMeters <= 0 || weight <= 0) return null
  return Number((weight / (heightMeters * heightMeters)).toFixed(1))
}

export function validateStep(step, form, bmi) {
  const errors = {}

  if (step === 0) {
    const age = Number(form.age)
    const height = Number(form.heightCm)
    const weight = Number(form.weightKg)
    if (!form.age || !Number.isInteger(age) || age < 18 || age > 100) {
      errors.age = 'Enter an age between 18 and 100.'
    }
    if (form.gender === '') errors.gender = 'Select a gender.'
    if (!form.heightCm || !Number.isFinite(height) || height <= 0 || height > 250) {
      errors.heightCm = 'Enter a height greater than 0 and no more than 250 cm.'
    }
    if (!form.weightKg || !Number.isFinite(weight) || weight <= 0 || weight > 300) {
      errors.weightKg = 'Enter a weight greater than 0 and no more than 300 kg.'
    }
    if (bmi !== null && (bmi <= 0 || bmi > 80)) {
      errors.weightKg = 'The calculated BMI must be between 0 and 80.'
    }
  }

  if (step === 1) {
    if (!Number.isFinite(Number(form.VAS_score)) || Number(form.VAS_score) < 0 || Number(form.VAS_score) > 10) {
      errors.VAS_score = 'Choose a pain level from 0 to 10.'
    }
    symptomFields.forEach(({ name, label }) => {
      if (form[name] === '') errors[name] = `Select an answer for ${label.toLowerCase()}.`
    })
  }

  if (step === 2) {
    historyFields.forEach(({ name, label }) => {
      if (form[name] === '') errors[name] = `Select an answer for ${label.toLowerCase()}.`
    })
  }

  return errors
}

export function nextStepAfterValidation(currentStep, errors) {
  if (Object.keys(errors).length > 0) return currentStep
  return Math.min(currentStep + 1, 2)
}
