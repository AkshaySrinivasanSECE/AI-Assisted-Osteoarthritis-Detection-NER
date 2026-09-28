import RiskGauge from './RiskGauge'

function ResultCard({ result, patient }) {
  const riskLabel = result.risk_level?.toUpperCase() || 'LOW'
  const parseGender = (gender) => {
    if (gender === 0) return 'Female'
    if (gender === 1) return 'Male'
    return 'Not specified'
  }

  return (
    <div className="result-layout">
      <div className="result-panel primary-result">
        <div className="panel-header row-between">
          <span className="section-kicker">OA Risk Assessment</span>
          <span className={`risk-badge risk-badge--${result.risk_level?.toLowerCase() || 'low'}`}>
            {riskLabel} RISK
          </span>
        </div>

        <RiskGauge
          oaProbability={result.oa_probability}
          healthyProbability={result.healthy_probability}
          riskLevel={result.risk_level}
        />

        <div className="assessment-card">
          <h3>AI Screening Assessment</h3>
          <p>{result.message}</p>
        </div>

        <div className="recommendation-box">
          <h4>Recommended Next Step</h4>
          <p>
            {result.risk_level === 'Low'
              ? 'Continue routine monitoring and seek clinical evaluation if symptoms persist.'
              : 'Consider further clinical evaluation by a qualified healthcare professional.'}
          </p>
        </div>
      </div>

      <aside className="result-panel summary-panel">
        <h3>Patient Summary</h3>
        <div className="summary-list">
          <div><span>Age</span><strong>{patient.age}</strong></div>
          <div><span>Gender</span><strong>{parseGender(patient.gender)}</strong></div>
          <div><span>BMI</span><strong>{Number(patient.BMI).toFixed(2)}</strong></div>
          <div><span>VAS Score</span><strong>{patient.VAS_score}</strong></div>
          <div><span>JPR 30°</span><strong>{patient.JPR_30}</strong></div>
          <div><span>JPR 45°</span><strong>{patient.JPR_45}</strong></div>
          <div><span>JPR 60°</span><strong>{patient.JPR_60}</strong></div>
        </div>
      </aside>
    </div>
  )
}

export default ResultCard
