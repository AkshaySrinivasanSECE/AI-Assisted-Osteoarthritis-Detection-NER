function RiskGauge({ oaProbability = 0, healthyProbability = 0, riskLevel = 'Low' }) {
  const safeOa = Number.isFinite(oaProbability) ? oaProbability : 0
  const safeHealthy = Number.isFinite(healthyProbability) ? healthyProbability : 0
  const displayedOa = safeOa.toFixed(1)
  const displayedHealthy = safeHealthy.toFixed(1)
  const circumference = 2 * Math.PI * 54
  const dashOffset = circumference - (safeOa / 100) * circumference

  return (
    <div className="risk-gauge-wrap">
      <div className="gauge-ring" aria-label={`OA probability ${displayedOa} percent`}>
        <svg viewBox="0 0 140 140" className="gauge-svg" role="img">
          <circle cx="70" cy="70" r="54" className="gauge-track" />
          <circle
            cx="70"
            cy="70"
            r="54"
            className={`gauge-progress ${riskLevel.toLowerCase()}`}
            style={{ strokeDasharray: circumference, strokeDashoffset: dashOffset }}
          />
        </svg>
        <div className="gauge-label">
          <span className="small-label">OA Probability</span>
          <strong>{displayedOa}%</strong>
        </div>
      </div>

      <div className="gauge-details">
        <div>
          <span className="label">Healthy Pattern</span>
          <strong>{displayedHealthy}%</strong>
        </div>
        <div>
          <span className="label">Risk Category</span>
          <strong>{riskLevel}</strong>
        </div>
      </div>
    </div>
  )
}

export default RiskGauge
