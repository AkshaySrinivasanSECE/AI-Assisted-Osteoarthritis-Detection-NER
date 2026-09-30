const stages = [
  'Processing patient information',
  'Preparing age, gender, BMI, and pain score',
  'Running the selected screening model',
  'Generating screening assessment',
]

function AnalysisLoader() {
  return (
    <div className="analysis-loader" aria-live="polite" aria-busy="true">
      <div className="loader-icon" aria-hidden="true">
        <span className="pulse-dot" />
      </div>

      <h3>Analyzing OA Risk Markers...</h3>

      <div className="loader-list">
        {stages.map((stage, index) => (
          <div key={stage} className="loader-item">
            <span className="loader-bullet">{index + 1}</span>
            <span>{stage}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default AnalysisLoader
