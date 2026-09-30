import { useEffect, useState } from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import FeatureCard from './components/FeatureCard'
import ScreeningForm from './components/ScreeningForm'
import ProgressSteps from './components/ProgressSteps'
import AnalysisLoader from './components/AnalysisLoader'
import ResultCard from './components/ResultCard'
import ModelInfo from './components/ModelInfo'
import Footer from './components/Footer'
import {
  buildPredictionPayload,
  fetchEvaluation,
  fetchModelInfo,
  submitRiskAssessment,
} from './services/api'
import './App.css'

const featureCards = [
  {
    icon: '01',
    title: 'Early Risk Identification',
    description: 'Identify patterns associated with OA risk before advanced deterioration.',
  },
  {
    icon: '02',
    title: 'Accessible Screening',
    description: 'Designed for simple clinical or healthcare-worker assisted screening.',
  },
  {
    icon: '03',
    title: 'AI-Assisted Decision Support',
    description: 'Supports healthcare professionals with data-driven risk assessment.',
  },
]

const futureScope = [
  'multilingual support',
  'offline-first screening',
  'portable deployment',
  'healthcare worker dashboard',
  'gait analysis',
  'sensor integration',
]

function App() {
  const [screeningView, setScreeningView] = useState('landing')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState('')
  const [result, setResult] = useState(null)
  const [patient, setPatient] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [evaluation, setEvaluation] = useState(null)
  const [modelDataLoading, setModelDataLoading] = useState(true)
  const [modelDataError, setModelDataError] = useState('')
  const workflowSteps = [
    'Patient Details',
    'Pain and Symptoms',
    'Automatic BMI',
    modelInfo?.features?.length
      ? `${modelInfo.features.length}-Feature AI Model`
      : 'AI Model',
    'Screening Result',
  ]

  const handleModelDataRetry = async () => {
    setModelDataLoading(true)
    setModelDataError('')

    try {
      const [modelInfoResponse, evaluationResponse] = await Promise.all([
        fetchModelInfo(),
        fetchEvaluation(),
      ])
      setModelInfo(modelInfoResponse)
      setEvaluation(evaluationResponse)
    } catch (error) {
      setModelDataError(error.message || 'Model information could not be loaded.')
    } finally {
      setModelDataLoading(false)
    }
  }

  useEffect(() => {
    let isCurrent = true

    Promise.all([fetchModelInfo(), fetchEvaluation()])
      .then(([modelInfoResponse, evaluationResponse]) => {
        if (!isCurrent) return
        setModelInfo(modelInfoResponse)
        setEvaluation(evaluationResponse)
      })
      .catch((error) => {
        if (isCurrent) {
          setModelDataError(error.message || 'Model information could not be loaded.')
        }
      })
      .finally(() => {
        if (isCurrent) setModelDataLoading(false)
      })

    return () => {
      isCurrent = false
    }
  }, [])

  const handleStartScreening = () => {
    setScreeningView('screening')
    setSubmitError('')
    setResult(null)
    setPatient(null)
    window.scrollTo({
      top: document.getElementById('screening')?.offsetTop || 0,
      behavior: 'smooth',
    })
  }

  const handleReset = () => {
    setScreeningView('landing')
    setResult(null)
    setPatient(null)
    setSubmitError('')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleSubmit = async (payload) => {
    setIsSubmitting(true)
    setSubmitError('')
    setPatient(payload)

    try {
      // Supporting answers stay in the patient summary and never enter the ML prediction.
      const response = await submitRiskAssessment(buildPredictionPayload(payload))
      setResult(response)
      setScreeningView('result')
    } catch (error) {
      setSubmitError(error.message || 'The screening service is not available right now.')
      setScreeningView('screening')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="app-shell">
      <Navbar onStartScreening={handleStartScreening} />

      <main>
        <Hero onStartScreening={handleStartScreening} modelInfo={modelInfo} />

        <section className="content-section">
          <div className="container section-header-block">
            <div className="section-kicker">Why this project</div>
            <h2>Human-centred, accessible risk screening</h2>
          </div>

          <div className="container feature-grid">
            {featureCards.map((card) => (
              <FeatureCard key={card.title} {...card} />
            ))}
          </div>
        </section>

        <section id="how-it-works" className="content-section">
          <div className="container section-header-block">
            <div className="section-kicker">How it works</div>
            <h2>From patient information to screening result</h2>
          </div>

          <div className="container workflow-wrap">
            <div className="workflow-steps" aria-label="AI screening workflow">
              {workflowSteps.map((step, index) => (
                <div key={step} className="workflow-step">
                  <div className="workflow-node">
                    <span>{index + 1}</span>
                  </div>
                  <div className="workflow-label">{step}</div>
                  {index < workflowSteps.length - 1 && <div className="workflow-line" aria-hidden="true" />}
                </div>
              ))}
            </div>
          </div>
        </section>

        {screeningView === 'screening' && !isSubmitting && (
          <ScreeningForm
            onSubmit={handleSubmit}
            isSubmitting={isSubmitting}
            submitError={submitError}
            onReset={handleReset}
          />
        )}

        {screeningView === 'result' && !isSubmitting && result && patient && (
          <section className="container result-section">
            <ProgressSteps activeStep={4} />
            <ResultCard result={result} patient={patient} modelInfo={modelInfo} />
          </section>
        )}

        {isSubmitting && (
          <section className="container loader-section">
            <ProgressSteps activeStep={3} />
            <AnalysisLoader />
          </section>
        )}

        <ModelInfo
          modelInfo={modelInfo}
          evaluation={evaluation}
          isLoading={modelDataLoading}
          error={modelDataError}
          onRetry={handleModelDataRetry}
        />

        <section id="about" className="content-section ner-section">
          <div className="container ner-layout">
            <div>
              <div className="section-kicker">NER Context</div>
              <h2>Built for accessible screening in underserved regions</h2>
              <p>
                Designed with the long-term goal of supporting accessible OA risk screening in underserved and remote areas of India&apos;s North Eastern Region.
              </p>
            </div>
            <div className="scope-card">
              <h3>Future scope</h3>
              <ul>
                {futureScope.map((scope) => (
                  <li key={scope}>{scope}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  )
}

export default App
