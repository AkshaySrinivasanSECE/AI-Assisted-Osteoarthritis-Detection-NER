function RiskGauge({ oaProbability = 0, healthyProbability = 0, riskLevel = 'Low' }) {
  const safeOa = Number.isFinite(oaProbability) ? oaProbability : 0
  const circumference = 2 * Math.PI * 54
  const dashOffset = circumference - (safeOa / 100) * circumference

  return (
    <div className="risk-gauge-wrap">
      <div className="gauge-ring" aria-label={`OA Probability ${safeOa}%`}>
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
          <strong>{safeOa.toFixed(2)}%</strong>
        </div>
      </div>

      <div className="gauge-details">
        <div>
          <span className="label">Healthy Pattern</span>
          <strong>{Number(healthyProbability || 0).toFixed(2)}%</strong>
        </div>
        <div className="risk-badge risk-badge--large risk-badge--status">
          {riskLevel.toUpperCase()} RISK
        </div>
      </div>
    </div>
  )
}

export default RiskGauge
