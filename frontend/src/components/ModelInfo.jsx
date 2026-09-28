const features = [
  'Age',
  'Gender',
  'BMI',
  'VAS Pain Score',
  'JPR 30°',
  'JPR 45°',
  'JPR 60°',
]

function ModelInfo() {
  return (
    <section id="ai-model" className="info-section">
      <div className="container section-header-block">
        <div className="section-kicker">About the AI Model</div>
        <h2>Clinical risk screening model</h2>
      </div>

      <div className="container model-grid">
        <div className="model-card">
          <h3>Model overview</h3>
          <ul className="detail-list">
            <li><span>Algorithm:</span> <strong>Random Forest Classifier</strong></li>
            <li><span>Training Data:</span> <strong>Knee osteoarthritis clinical dataset</strong></li>
            <li><span>Output:</span> <strong>OA risk screening classification and probability</strong></li>
          </ul>
        </div>

        <div className="model-card">
          <h3>Input features</h3>
          <ul className="chip-list">
            {features.map((feature) => (
              <li key={feature}>{feature}</li>
            ))}
          </ul>
        </div>

        <div className="model-card wide-card">
          <h3>Models evaluated</h3>
          <div className="model-evaluated-list">
            <span>Logistic Regression</span>
            <span>KNN</span>
            <span>Decision Tree</span>
            <span>Random Forest</span>
            <span>SVM</span>
          </div>
          <p>
            Random Forest was selected because it produced the strongest F1 performance during our current
            cross-validation experiment.
          </p>
        </div>
      </div>
    </section>
  )
}

export default ModelInfo
