const FEATURE_LABELS = {
  age: 'Age',
  gender: 'Gender',
  BMI: 'BMI',
  'VAS score': 'VAS Pain Score',
}

const METRICS = [
  ['accuracy', 'Accuracy'],
  ['precision', 'Precision'],
  ['recall', 'Recall'],
  ['f1', 'F1'],
]

function formatMetric(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Not available'
}

function formatMetricWithStd(value, standardDeviation) {
  if (!Number.isFinite(value)) return 'Not available'
  if (!Number.isFinite(standardDeviation)) return formatMetric(value)
  return `${formatMetric(value)} ± ${(standardDeviation * 100).toFixed(1)}%`
}

function formatTimestamp(timestamp) {
  if (!timestamp) return 'Not recorded'
  const parsed = new Date(timestamp)
  return Number.isNaN(parsed.getTime()) ? timestamp : parsed.toLocaleString()
}

function directionLabel(direction) {
  if (direction === 'toward_oa') return 'Higher values lean toward OA'
  if (direction === 'toward_healthy') return 'Higher values lean toward healthy'
  return 'No directional effect'
}

function LoadingState() {
  return (
    <div className="container model-data-state" role="status" aria-live="polite">
      <span className="pulse-dot" aria-hidden="true" />
      <div>
        <h3>Loading model evidence</h3>
        <p>Retrieving current model metadata and evaluation results from the API.</p>
      </div>
    </div>
  )
}

function ErrorState({ message, onRetry }) {
  return (
    <div className="container model-data-state model-data-state--error" role="alert">
      <div>
        <h3>Model evidence is temporarily unavailable</h3>
        <p>{message}</p>
      </div>
      <button type="button" className="secondary-button compact-button" onClick={onRetry}>
        Retry loading
      </button>
    </div>
  )
}

function ConfusionMatrix({ confusionMatrix }) {
  const { labels = [], matrix = [], evaluated_on: evaluatedOn } = confusionMatrix || {}
  const flatValues = matrix.flat().filter(Number.isFinite)
  const maximum = Math.max(...flatValues, 1)

  if (matrix.length !== 2 || labels.length !== 2) return null

  return (
    <article className="model-card evaluation-card">
      <div className="card-heading-row">
        <div>
          <span className="card-eyebrow">Classification outcomes</span>
          <h3>Confusion matrix</h3>
        </div>
        <span className="evidence-tag">Unified active-data validation</span>
      </div>
      <p className="card-description">Rows are actual labels; columns are model predictions.</p>

      <div className="confusion-matrix-wrap">
        <div className="matrix-axis matrix-axis--column" aria-hidden="true">Predicted label</div>
        <div className="matrix-axis matrix-axis--row" aria-hidden="true">Actual label</div>
        <div
          className="confusion-grid"
          role="img"
          aria-label={`Confusion matrix evaluated on ${evaluatedOn}. Actual ${labels[0]}: ${matrix[0][0]} predicted ${labels[0]} and ${matrix[0][1]} predicted ${labels[1]}. Actual ${labels[1]}: ${matrix[1][0]} predicted ${labels[0]} and ${matrix[1][1]} predicted ${labels[1]}.`}
        >
          <span aria-hidden="true" />
          {labels.map((label) => <strong key={`column-${label}`} className="matrix-label">{label}</strong>)}
          {matrix.map((row, rowIndex) => (
            <div className="matrix-row" key={labels[rowIndex]}>
              <strong className="matrix-label matrix-label--row">{labels[rowIndex]}</strong>
              {row.map((value, columnIndex) => (
                <div
                  key={`${rowIndex}-${columnIndex}`}
                  className="matrix-cell"
                  style={{ '--cell-intensity': Math.max(value / maximum, 0.14) }}
                  title={`Actual ${labels[rowIndex]}, predicted ${labels[columnIndex]}: ${value}`}
                >
                  <strong>{value}</strong>
                  <span>{rowIndex === columnIndex ? 'Correct' : 'Incorrect'}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      <p className="evidence-note">Evaluated on {evaluatedOn}.</p>
    </article>
  )
}

function FeatureImportance({ features, method }) {
  if (!features?.length) return null

  return (
    <article className="model-card evaluation-card">
      <div className="card-heading-row">
        <div>
          <span className="card-eyebrow">Model-based explanation</span>
          <h3>Global feature importance</h3>
        </div>
      </div>
      <p className="card-description">{method}</p>
      <div className="importance-chart" role="list" aria-label="Global feature importance">
        {features.map((item) => {
          const percentage = Math.max(0, Math.min(100, item.relative_importance))
          return (
            <div className="importance-row" role="listitem" key={item.feature}>
              <div className="importance-label-row">
                <strong>{FEATURE_LABELS[item.feature] || item.feature}</strong>
                <span>{percentage.toFixed(1)}%</span>
              </div>
              <div
                className="importance-track"
                role="img"
                aria-label={`${FEATURE_LABELS[item.feature] || item.feature}: ${percentage.toFixed(1)} percent relative importance. ${directionLabel(item.direction_when_feature_increases)}.`}
              >
                <span style={{ width: `${percentage}%` }} />
              </div>
              <small>{directionLabel(item.direction_when_feature_increases)}</small>
            </div>
          )
        })}
      </div>
      <p className="evidence-note">Importance describes the fitted model, not a causal effect or clinical risk factor.</p>
    </article>
  )
}

function ModelInfo({ modelInfo, evaluation, isLoading, error, onRetry }) {
  const selectedMetrics = evaluation?.candidate_models?.find(
    (candidate) => candidate.model === evaluation.selected_model,
  )
  const validation = modelInfo?.validation_strategy

  return (
    <section id="ai-model" className="info-section" aria-labelledby="model-section-title">
      <div className="container section-header-block">
        <div className="section-kicker">About the AI Model</div>
        <h2 id="model-section-title">Evidence behind this screening result</h2>
        <p className="section-introduction">
          Current model details are loaded from the same backend that produces predictions.
        </p>
      </div>

      {isLoading && <LoadingState />}
      {!isLoading && error && <ErrorState message={error} onRetry={onRetry} />}

      {!isLoading && !error && modelInfo && evaluation && (
        <div className="container model-evidence-stack">
          <div className="model-overview-grid">
            <article className="model-card model-overview-card">
              <div className="card-heading-row">
                <div>
                  <span className="card-eyebrow">AI model overview</span>
                  <h3>{modelInfo.selected_model}</h3>
                </div>
                <span className="model-version">{modelInfo.model_version}</span>
              </div>
              <ul className="detail-list">
                <li><span>Prediction task</span><strong>OA risk screening classification and probability</strong></li>
                <li><span>Selection rule</span><strong>Highest mean F1 across candidate models</strong></li>
                <li><span>Training timestamp</span><strong>{formatTimestamp(modelInfo.training_timestamp)}</strong></li>
              </ul>
            </article>

            <article className="model-card">
              <span className="card-eyebrow">Model inputs</span>
              <h3>{modelInfo.features.length} features used</h3>
              <ul className="chip-list" aria-label="Model input features">
                {modelInfo.features.map((feature) => (
                  <li key={feature}>{FEATURE_LABELS[feature] || feature}</li>
                ))}
              </ul>
              <p className="card-description">Supporting symptom and history answers do not enter the current model.</p>
            </article>
          </div>

          <section aria-labelledby="dataset-title">
            <div className="subsection-heading">
              <div>
                <span className="card-eyebrow">Dataset</span>
                <h3 id="dataset-title">Unified model training dataset</h3>
              </div>
            </div>
            <div className="provenance-grid">
              <article className="provenance-card provenance-card--total">
                <span className="provenance-type">Training data</span>
                <strong>{modelInfo.total_training_rows}</strong>
                <h4>Rows used by the model</h4>
                <p>The model is trained and evaluated using one unified dataset.</p>
              </article>
            </div>
          </section>

          <article className="model-card validation-card">
            <div>
              <span className="card-eyebrow">Validation methodology</span>
              <h3>{validation?.stratified ? 'Stratified ' : ''}{validation?.folds}-fold cross-validation</h3>
              <p>{evaluation.validation_strategy}</p>
            </div>
            <dl className="validation-facts">
              <div><dt>Validation subjects</dt><dd>{validation?.validation_subjects}</dd></div>
              <div><dt>Validation rows</dt><dd>{validation?.validation_subjects}</dd></div>
              <div><dt>Random seed</dt><dd>{validation?.random_seed}</dd></div>
              <div><dt>Shuffle</dt><dd>{validation?.shuffle ? 'Enabled' : 'Disabled'}</dd></div>
            </dl>
          </article>

          {selectedMetrics && (
            <section aria-labelledby="metrics-title">
              <div className="subsection-heading">
                <div>
                  <span className="card-eyebrow">Evaluation metrics</span>
                  <h3 id="metrics-title">Selected model performance</h3>
                </div>
                <span className="evidence-tag">Mean ± standard deviation</span>
              </div>
              <div className="metrics-grid">
                {METRICS.map(([key, label]) => (
                  <article className="metric-card" key={key}>
                    <span>{label}</span>
                    <strong>{formatMetric(selectedMetrics[key])}</strong>
                    <small>SD {formatMetric(selectedMetrics[`${key}_std`])}</small>
                  </article>
                ))}
              </div>
            </section>
          )}

          <section className="model-card comparison-card" aria-labelledby="comparison-title">
            <div className="card-heading-row">
              <div>
                <span className="card-eyebrow">Candidate evaluation</span>
                <h3 id="comparison-title">Model comparison</h3>
              </div>
              <span className="evidence-tag">Selected by mean F1</span>
            </div>
            <div className="table-scroll" tabIndex="0" aria-label="Scrollable model comparison table">
              <table className="comparison-table">
                <caption className="sr-only">Cross-validation means and standard deviations for all candidate models</caption>
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    {METRICS.map(([, label]) => <th scope="col" key={label}>{label}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {evaluation.candidate_models.map((candidate) => {
                    const isSelected = candidate.model === evaluation.selected_model
                    return (
                      <tr key={candidate.model} className={isSelected ? 'selected-model-row' : ''}>
                        <th scope="row">
                          {candidate.model}
                          {isSelected && <span className="selected-label">Selected</span>}
                        </th>
                        {METRICS.map(([key]) => (
                          <td key={key}>{formatMetricWithStd(candidate[key], candidate[`${key}_std`])}</td>
                        ))}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>

          <div className="evaluation-visual-grid">
            <ConfusionMatrix confusionMatrix={evaluation.confusion_matrix} />
            <FeatureImportance
              features={evaluation.global_feature_importance}
              method={evaluation.global_feature_importance_method}
            />
          </div>

          <div className="evaluation-disclaimer" role="note">
            <strong>Evidence boundary</strong>
            <p>{evaluation.disclaimer}</p>
          </div>
        </div>
      )}
    </section>
  )
}

export default ModelInfo
