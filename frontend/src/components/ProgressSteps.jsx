const steps = [
  'Patient Details',
  'Symptoms',
  'Medical History',
  'AI Analysis',
  'Result',
]

function ProgressSteps({ activeStep = 0 }) {
  return (
    <div className="progress-steps" aria-label="Screening progress">
      {steps.map((step, index) => {
        const isActive = index === activeStep
        const isComplete = index < activeStep

        return (
          <div key={step} className={`progress-step ${isActive ? 'active' : ''} ${isComplete ? 'complete' : ''}`}>
            <span className="step-bullet">{index + 1}</span>
            <span className="step-label">{step}</span>
            {index < steps.length - 1 && <span className="step-separator">→</span>}
          </div>
        )
      })}
    </div>
  )
}

export default ProgressSteps
