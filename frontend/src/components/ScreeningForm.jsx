import { useMemo, useState } from 'react'
import ProgressSteps from './ProgressSteps'
import PainSlider from './PainSlider'
import {
  calculateBMI,
  historyFields,
  nextStepAfterValidation,
  symptomFields,
  validateStep,
} from '../utils/screeningLogic'

const defaultForm = {
  age: '',
  gender: '',
  heightCm: '',
  weightKg: '',
  VAS_score: 5,
  affectedKnee: '',
  morningStiffness: '',
  swelling: '',
  walkingDifficulty: '',
  stairDifficulty: '',
  standingDifficulty: '',
  previousInjury: '',
  previousSurgery: '',
  familyHistory: '',
  activityLevel: '',
  occupationType: '',
}

function ScreeningForm({ onSubmit, isSubmitting, submitError, onReset }) {
  const [form, setForm] = useState(defaultForm)
  const [step, setStep] = useState(0)
  const [errors, setErrors] = useState({})

  const calculatedBMI = useMemo(() => {
    return calculateBMI(form.heightCm, form.weightKg)
  }, [form.heightCm, form.weightKg])

  const handleFieldChange = (name, value) => {
    setForm((previous) => ({ ...previous, [name]: value }))
    setErrors((previous) => ({ ...previous, [name]: '' }))
  }

  const handleNext = () => {
    const nextErrors = validateStep(step, form, calculatedBMI)
    setErrors(nextErrors)
    setStep((current) => nextStepAfterValidation(current, nextErrors))
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    const stepErrors = [0, 1, 2].map((stepIndex) => validateStep(stepIndex, form, calculatedBMI))
    const firstInvalidStep = stepErrors.findIndex((stepErrorsForStep) => Object.keys(stepErrorsForStep).length > 0)
    if (firstInvalidStep !== -1) {
      setErrors(stepErrors[firstInvalidStep])
      setStep(firstInvalidStep)
      return
    }

    onSubmit({
      ...form,
      age: Number(form.age),
      gender: Number(form.gender),
      heightCm: Number(form.heightCm),
      weightKg: Number(form.weightKg),
      BMI: calculatedBMI,
      VAS_score: Number(form.VAS_score),
    })
  }

  const renderSelect = ({ name, label, options }) => (
    <div className="field-card" key={name}>
      <label htmlFor={name}>{label}</label>
      <select
        id={name}
        value={form[name]}
        onChange={(event) => handleFieldChange(name, event.target.value)}
        aria-invalid={Boolean(errors[name])}
      >
        <option value="">Select an option</option>
        {options.map((option) => <option key={option} value={option}>{option}</option>)}
      </select>
      {errors[name] && <span className="error-text">{errors[name]}</span>}
    </div>
  )

  return (
    <section id="screening" className="screening-section">
      <div className="container screening-shell">
        <div className="screening-header row-between">
          <div>
            <div className="section-kicker">New OA Screening</div>
            <h2>Patient screening</h2>
          </div>
          <button type="button" className="secondary-button compact-button" onClick={onReset}>
            Exit screening
          </button>
        </div>

        <ProgressSteps activeStep={step} />

        <form className="screening-form" onSubmit={handleSubmit} noValidate>
          {step === 0 && (
            <div className="form-section">
              <div className="section-title-row">
                <div>
                  <span className="form-section-label">AI MODEL INPUTS</span>
                  <h3>Basic details</h3>
                  <p className="model-input-copy">
                    The current AI model uses age, gender, BMI, and pain score for prediction.
                  </p>
                </div>
              </div>
              <div className="field-grid two-up">
                <div className="field-card">
                  <label htmlFor="age">Age</label>
                  <input id="age" type="number" min="18" max="100" step="1" value={form.age}
                    onChange={(event) => handleFieldChange('age', event.target.value)} aria-invalid={Boolean(errors.age)} />
                  {errors.age && <span className="error-text">{errors.age}</span>}
                </div>
                <div className="field-card">
                  <label htmlFor="gender">Gender</label>
                  <select id="gender" value={form.gender}
                    onChange={(event) => handleFieldChange('gender', event.target.value)} aria-invalid={Boolean(errors.gender)}>
                    <option value="">Select an option</option>
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                  </select>
                  <small>Choose the option matching the dataset categories.</small>
                  {errors.gender && <span className="error-text">{errors.gender}</span>}
                </div>
                <div className="field-card">
                  <label htmlFor="heightCm">Height (cm)</label>
                  <input id="heightCm" type="number" min="1" max="250" step="0.1" value={form.heightCm}
                    onChange={(event) => handleFieldChange('heightCm', event.target.value)} aria-invalid={Boolean(errors.heightCm)} />
                  {errors.heightCm && <span className="error-text">{errors.heightCm}</span>}
                </div>
                <div className="field-card">
                  <label htmlFor="weightKg">Weight (kg)</label>
                  <input id="weightKg" type="number" min="1" max="300" step="0.1" value={form.weightKg}
                    onChange={(event) => handleFieldChange('weightKg', event.target.value)} aria-invalid={Boolean(errors.weightKg)} />
                  {errors.weightKg && <span className="error-text">{errors.weightKg}</span>}
                </div>
              </div>
              <div className="bmi-readout" aria-live="polite">
                <span>Calculated BMI</span>
                <strong>{calculatedBMI === null ? '--' : calculatedBMI.toFixed(1)}</strong>
                <small>Calculated from height and weight; BMI is sent to the model.</small>
              </div>
            </div>
          )}

          {step === 1 && (
            <>
              <div className="form-section">
                <div className="section-title-row">
                  <div>
                    <span className="form-section-label">AI MODEL INPUTS</span>
                    <h3>Pain score</h3>
                    <p className="model-input-copy">
                      The current AI model uses age, gender, BMI, and pain score for prediction.
                    </p>
                  </div>
                </div>
                <PainSlider value={form.VAS_score} onChange={(value) => handleFieldChange('VAS_score', value)} />
                {errors.VAS_score && <span className="error-text">{errors.VAS_score}</span>}
              </div>
              <div className="form-section">
                <div className="section-title-row">
                  <div>
                    <span className="form-section-label">ADDITIONAL SCREENING INFORMATION</span>
                    <h3>Symptoms</h3>
                    <p className="supporting-data-note">
                      These answers provide screening context and do not influence the current AI prediction.
                    </p>
                  </div>
                </div>
                <div className="field-grid two-up">{symptomFields.map(renderSelect)}</div>
              </div>
            </>
          )}

          {step === 2 && (
            <div className="form-section">
              <div className="section-title-row">
                <div>
                  <span className="form-section-label">ADDITIONAL SCREENING INFORMATION</span>
                  <h3>Medical history and activity</h3>
                  <p className="supporting-data-note">
                    These answers provide screening context and do not influence the current AI prediction.
                  </p>
                </div>
              </div>
              <div className="field-grid two-up">{historyFields.map(renderSelect)}</div>
            </div>
          )}

          {submitError && <div className="submit-error" role="alert">{submitError}</div>}

          <div className="form-actions">
            {step > 0 && <button type="button" className="secondary-button" onClick={() => setStep((current) => current - 1)}>Back</button>}
            {step < 2 ? (
              <button type="button" className="primary-button large-button" onClick={handleNext}>Continue</button>
            ) : (
              <button type="submit" className="primary-button large-button" disabled={isSubmitting}>
                {isSubmitting ? 'Analyzing...' : 'Analyze OA Risk'}
              </button>
            )}
          </div>
        </form>
      </div>
    </section>
  )
}

export default ScreeningForm
