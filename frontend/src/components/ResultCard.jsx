import RiskGauge from './RiskGauge'

const FEATURE_LABELS = {
  age: 'Age',
  gender: 'Gender',
  BMI: 'BMI',
  'VAS score': 'VAS Pain Score',
  VAS_score: 'VAS Pain Score',
}

const RESULT_STAGES = ['Input', 'AI Model', 'Prediction', 'Explanation', 'Next Step']

function parseGender(gender) {
  if (Number(gender) === 0) return 'Female'
  if (Number(gender) === 1) return 'Male'
  return 'Not specified'
}

function formatInput(feature, value) {
  if (feature === 'gender') return parseGender(value)
  if (feature === 'age') return `${value} years`
  if (feature === 'BMI') return Number(value).toFixed(1)
  if (feature === 'VAS score' || feature === 'VAS_score') return `${value} / 10`
  return value
}

function directionText(direction) {
  if (direction === 'toward_oa') return 'Leans toward OA pattern'
  if (direction === 'toward_healthy') return 'Leans toward healthy pattern'
  return 'Neutral contribution'
}

function ResultFlow() {
  return (
    <nav className="result-flow" aria-label="Prediction result stages">
      <ol>
        {RESULT_STAGES.map((stage, index) => (
          <li key={stage}>
            <span className="result-flow-number" aria-hidden="true">{index + 1}</span>
            <strong>{stage}</strong>
            {index < RESULT_STAGES.length - 1 && (
              <span className="result-flow-arrow" aria-hidden="true">↓</span>
            )}
          </li>
        ))}
      </ol>
    </nav>
  )
}

function ModelInputs({ values }) {
  const orderedFeatures = ['age', 'gender', 'BMI', 'VAS_score']
    .filter((feature) => Object.hasOwn(values, feature))

  return (
    <section className="result-panel result-input-panel" aria-labelledby="model-inputs-title">
      <span className="result-stage-label">Input</span>
      <h2 id="model-inputs-title">Values used by the model</h2>
      <p>Only these {orderedFeatures.length} inputs influenced this prediction.</p>
      <dl className="result-input-grid">
        {orderedFeatures.map((feature) => (
          <div key={feature}>
            <dt>{FEATURE_LABELS[feature]}</dt>
            <dd>{formatInput(feature, values[feature])}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

function PredictionExplanation({ explanation }) {
  const contributions = explanation?.local_feature_contributions || []

  return (
    <section className="result-panel result-explanation-panel" aria-labelledby="prediction-explanation-title">
      <span className="result-stage-label">Explanation</span>
      <h2 id="prediction-explanation-title">Why did the model produce this result?</h2>
      <p className="result-section-intro">
        Each bar shows its share of the model&apos;s total absolute contribution for this prediction.
        Direction describes the model output, not cause and effect.
      </p>

      {contributions.length > 0 ? (
        <div className="result-contribution-grid" role="list">
          {contributions.map((item) => {
            const percentage = Math.round(
              Math.max(0, Math.min(100, item.relative_contribution)),
            )
            const directionClass = item.contribution_direction.replaceAll('_', '-')

            return (
              <article className="result-contribution-card" role="listitem" key={item.feature}>
                <div className="result-contribution-header">
                  <div>
                    <strong>{FEATURE_LABELS[item.feature] || item.feature}</strong>
                    <span>Input: {formatInput(item.feature, item.input_value)}</span>
                  </div>
                  <strong className="result-contribution-percent">{percentage}%</strong>
                </div>
                <div
                  className={`contribution-track contribution-track--${directionClass}`}
                  role="img"
                  aria-label={`${FEATURE_LABELS[item.feature] || item.feature}: ${percentage} percent relative contribution. ${directionText(item.contribution_direction)}.`}
                >
                  <span style={{ width: `${percentage}%` }} />
                </div>
                <span className={`direction-label direction-label--${directionClass}`}>
                  {directionText(item.contribution_direction)}
                </span>
              </article>
            )
          })}
        </div>
      ) : (
        <p className="result-unavailable" role="status">
          Feature contribution details are unavailable for this prediction.
        </p>
      )}

      {explanation?.disclaimer && (
        <small className="explanation-disclaimer">{explanation.disclaimer}</small>
      )}
    </section>
  )
}

function ModelInformation({ result, modelInfo }) {
  const validation = modelInfo?.validation_strategy
  const validationLabel = validation
    ? `${validation.stratified ? 'Stratified ' : ''}${validation.folds}-fold cross-validation; unified dataset`
    : 'Model validation metadata is currently unavailable'

  return (
    <section className="result-panel result-model-panel" aria-labelledby="result-model-title">
      <span className="result-stage-label">AI Model</span>
      <div className="result-title-row">
        <h2 id="result-model-title">Model information</h2>
        <span className="model-version">{result.model_version}</span>
      </div>
      <dl className="result-model-grid">
        <div>
          <dt>Selected model</dt>
          <dd>{result.model_name}</dd>
        </div>
        <div>
          <dt>Validation strategy</dt>
          <dd>{validationLabel}</dd>
        </div>
        <div>
          <dt>Dataset rows</dt>
          <dd>{modelInfo?.original_subject_count ?? 'Unavailable'}</dd>
        </div>
        <div>
          <dt>Training rows</dt>
          <dd>{modelInfo?.total_training_rows ?? 'Unavailable'}</dd>
          <small>All rows are used as one unified model dataset.</small>
        </div>
      </dl>
    </section>
  )
}

function ResultCard({ result, patient, modelInfo }) {
  const riskLevel = result.risk_level || 'Low'
  const modelInputs = result.feature_values_used || {
    age: patient.age,
    gender: patient.gender,
    BMI: patient.BMI,
    VAS_score: patient.VAS_score,
  }

  const nextStep = riskLevel === 'Low'
    ? 'Monitor symptoms and seek professional assessment if knee pain, stiffness, or mobility difficulty persists or worsens.'
    : 'Consider discussing these screening results and your symptoms with a qualified healthcare professional for appropriate assessment.'

  return (
    <div className="result-dashboard">
      <ResultFlow />

      <section className="result-panel result-screening-panel" aria-labelledby="screening-result-title">
        <div className="result-title-row">
          <div>
            <span className="result-stage-label">Prediction</span>
            <h2 id="screening-result-title">AI Screening Result</h2>
          </div>
          <span className={`risk-badge risk-badge--${riskLevel.toLowerCase()}`}>
            {riskLevel.toUpperCase()} RISK
          </span>
        </div>

        <div className="result-screening-content">
          <RiskGauge
            oaProbability={result.oa_probability}
            healthyProbability={result.healthy_probability}
            riskLevel={riskLevel}
          />
          <div className="result-interpretation">
            <span>Screening interpretation</span>
            <p>{result.message}</p>
            <small>Risk category is based on the model&apos;s probability thresholds and is not a diagnosis.</small>
          </div>
        </div>
      </section>

      <ModelInputs values={modelInputs} />
      <PredictionExplanation explanation={result.explanation} />

      <ModelInformation result={result} modelInfo={modelInfo} />

      <section className="result-panel result-next-step" aria-labelledby="next-step-title">
        <span className="result-stage-label">Next Step</span>
        <h2 id="next-step-title">What to do next</h2>
        <p>{nextStep}</p>
      </section>

      <section className="result-important" aria-labelledby="important-title">
        <div className="important-mark" aria-hidden="true">!</div>
        <div>
          <h2 id="important-title">Important</h2>
          <p>{result.disclaimer || 'This is an AI-assisted screening prototype and not a medical diagnosis.'}</p>
        </div>
      </section>
    </div>
  )
}

export default ResultCard
