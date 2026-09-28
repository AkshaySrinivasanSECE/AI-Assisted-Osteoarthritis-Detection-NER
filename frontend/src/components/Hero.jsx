function Hero({ onStartScreening }) {
  return (
    <section id="home" className="hero-section">
      <div className="container hero-grid">
        <div className="hero-copy">
          <div className="eyebrow">AI-assisted screening platform</div>
          <h1>
            AI-Assisted Early Detection of
            <span>Osteoarthritis Risk</span>
          </h1>
          <p className="hero-subtitle">
            An intelligent screening platform designed to assist early identification of knee
            osteoarthritis risk markers, with a focus on accessible healthcare in the North Eastern Region.
          </p>
          <div className="hero-actions">
            <button type="button" className="primary-button" onClick={onStartScreening}>
              Start Screening
            </button>
            <a href="#how-it-works" className="secondary-button">
              How It Works
            </a>
          </div>
        </div>

        <div className="hero-visual" aria-label="AI medical screening visual">
          <div className="visual-glow" />
          <div className="joint-illustration">
            <div className="joint-knee">
              <span className="joint-core" />
              <span className="joint-ring ring-one" />
              <span className="joint-ring ring-two" />
            </div>
          </div>

          <div className="floating-card card-top">
            <span className="card-label">AI Scan</span>
            <strong>Risk markers</strong>
            <small>Joint pattern analysis</small>
          </div>

          <div className="floating-card card-bottom">
            <span className="card-label">Confidence</span>
            <strong>75.25%</strong>
            <small>OA probability</small>
          </div>

          <div className="signal-grid">
            <span />
            <span />
            <span />
            <span />
            <span />
            <span />
          </div>
        </div>
      </div>

      <div className="container stats-strip" aria-label="Project highlights">
        {[
          'AI-Assisted Screening',
          '7 Clinical Inputs',
          'Random Forest Model',
          'Real-Time Prediction',
        ].map((item) => (
          <div key={item} className="stat-item">
            {item}
          </div>
        ))}
      </div>
    </section>
  )
}

export default Hero
