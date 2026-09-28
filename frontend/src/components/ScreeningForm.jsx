import { useMemo, useState } from 'react'
import ProgressSteps from './ProgressSteps'
import PainSlider from './PainSlider'
import JPRInput from './JPRInput'

const defaultForm = {
  age: '',
  gender: 0,
  BMI: '',
  VAS_score: 5,
  JPR_30: '',
  JPR_45: '',
  JPR_60: '',
}

function ScreeningForm({ onSubmit, isSubmitting, submitError, onReset }) {
  const [form, setForm] = useState(defaultForm)
  const [touched, setTouched] = useState({})

  const errors = useMemo(() => {
    const nextErrors = {}

    if (!form.age && form.age !== 0) nextErrors.age = 'Age is required.'
    if (Number(form.age) < 18 || Number(form.age) > 100) {
      nextErrors.age = 'Age should be between 18 and 100.'
    }

    if (!form.BMI && form.BMI !== 0) nextErrors.BMI = 'BMI is required.'
    if (Number(form.BMI) <= 0 || Number(form.BMI) > 80) {
      nextErrors.BMI = 'BMI should be between 0 and 80.'
    }

    if (!form.JPR_30 && form.JPR_30 !== 0) nextErrors.JPR_30 = 'JPR 30° is required.'
    if (!form.JPR_45 && form.JPR_45 !== 0) nextErrors.JPR_45 = 'JPR 45° is required.'
    if (!form.JPR_60 && form.JPR_60 !== 0) nextErrors.JPR_60 = 'JPR 60° is required.'

    return nextErrors
  }, [form])

  const handleFieldChange = (name, value) => {
    setTouched((prev) => ({ ...prev, [name]: true }))
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = (event) => {
    event.preventDefault()

    const shouldBlock = Object.keys(errors).length > 0
    if (shouldBlock) {
      setTouched({
        age: true,
        gender: true,
        BMI: true,
        JPR_30: true,
        JPR_45: true,
        JPR_60: true,
      })
      return
    }

    const payload = {
      age: Number(form.age),
      gender: Number(form.gender),
      BMI: Number(form.BMI),
      VAS_score: Number(form.VAS_score),
      JPR_30: Number(form.JPR_30),
      JPR_45: Number(form.JPR_45),
      JPR_60: Number(form.JPR_60),
    }

    onSubmit(payload)
  }

  return (
    <section id="screening" className="screening-section">
      <div className="container screening-shell">
        <div className="screening-header row-between">
          <div>
            <div className="section-kicker">New OA Screening</div>
            <h2>Structured clinical assessment</h2>
          </div>
          <button type="button" className="secondary-button compact-button" onClick={onReset}>
            Reset
          </button>
        </div>

        <ProgressSteps activeStep={0} />

        <form className="screening-form" onSubmit={handleSubmit} noValidate>
          <div className="form-section">
            <div className="section-title-row">
              <h3>Patient Information</h3>
            </div>

            <div className="field-grid">
              <div className="field-card">
                <label htmlFor="age">Age</label>
                <input
                  id="age"
                  name="age"
                  type="number"
                  min="18"
                  max="100"
                  value={form.age}
                  onChange={(event) => handleFieldChange('age', event.target.value)}
                  placeholder="e.g. 60"
                  className={touched.age && errors.age ? 'error' : ''}
                />
                <small>Patient age in years.</small>
                {touched.age && errors.age && <span className="error-text">{errors.age}</span>}
              </div>

              <div className="field-card">
                <label htmlFor="gender">Gender</label>
                <select
                  id="gender"
                  name="gender"
                  value={form.gender}
                  onChange={(event) => handleFieldChange('gender', Number(event.target.value))}
                >
                  <option value={0}>Female</option>
                  <option value={1}>Male</option>
                </select>
                <small>Binary sex indicator used by the model.</small>
              </div>

              <div className="field-card">
                <label htmlFor="BMI">BMI</label>
                <input
                  id="BMI"
                  name="BMI"
                  type="number"
                  min="0"
                  max="80"
                  step="0.1"
                  value={form.BMI}
                  onChange={(event) => handleFieldChange('BMI', event.target.value)}
                  placeholder="e.g. 26.7"
                  className={touched.BMI && errors.BMI ? 'error' : ''}
                />
                <small>Body mass index.</small>
                {touched.BMI && errors.BMI && <span className="error-text">{errors.BMI}</span>}
              </div>
            </div>
          </div>

          <div className="form-section">
            <div className="section-title-row">
              <h3>Pain Assessment</h3>
            </div>
            <PainSlider value={form.VAS_score} onChange={(value) => handleFieldChange('VAS_score', value)} />
          </div>

          <div className="form-section">
            <div className="section-title-row">
              <h3>Joint Position Reproduction</h3>
            </div>
            <div className="field-grid three-up">
              <JPRInput
                label="JPR 30°"
                name="JPR_30"
                value={form.JPR_30}
                onChange={handleFieldChange}
                tooltip="Joint Position Reproduction measures how accurately the knee can reproduce a target joint angle and can provide information related to proprioceptive function."
              />
              <JPRInput
                label="JPR 45°"
                name="JPR_45"
                value={form.JPR_45}
                onChange={handleFieldChange}
                tooltip="Joint Position Reproduction measures how accurately the knee can reproduce a target joint angle and can provide information related to proprioceptive function."
              />
              <JPRInput
                label="JPR 60°"
                name="JPR_60"
                value={form.JPR_60}
                onChange={handleFieldChange}
                tooltip="Joint Position Reproduction measures how accurately the knee can reproduce a target joint angle and can provide information related to proprioceptive function."
              />
            </div>
          </div>

          {submitError && <div className="submit-error">{submitError}</div>}

          <button type="submit" className="primary-button large-button" disabled={isSubmitting}>
            {isSubmitting ? 'Analyzing...' : 'Analyze OA Risk'}
          </button>
        </form>
      </div>
    </section>
  )
}

export default ScreeningForm
