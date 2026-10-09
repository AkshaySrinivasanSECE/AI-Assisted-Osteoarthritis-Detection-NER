import assert from 'node:assert/strict'
import { after, before, describe, test } from 'node:test'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { createServer } from 'vite'

import {
  calculateBMI,
  nextStepAfterValidation,
  validateStep,
} from '../src/utils/screeningLogic.js'

const TEST_DIR = path.dirname(fileURLToPath(import.meta.url))
const FRONTEND_ROOT = path.resolve(TEST_DIR, '..')

let vite
let ScreeningForm
let AnalysisLoader
let ModelInfo
let ResultCard
let buildPredictionPayload

const completeForm = {
  age: '60',
  gender: '0',
  heightCm: '165',
  weightKg: '68',
  VAS_score: 5,
  affectedKnee: 'Left',
  morningStiffness: 'No',
  swelling: 'No',
  walkingDifficulty: 'Mild',
  stairDifficulty: 'Mild',
  standingDifficulty: 'None',
  previousInjury: 'No',
  previousSurgery: 'No',
  familyHistory: 'No',
  activityLevel: 'Moderate',
  occupationType: 'Mostly sitting',
}

const modelInfo = {
  selected_model: 'Logistic Regression',
  features: ['age', 'gender', 'BMI', 'VAS score'],
  original_subject_count: 88,
  synthetic_training_count: 500,
  total_training_rows: 588,
  model_version: 'oa-model-test',
  training_timestamp: null,
  validation_strategy: {
    name: 'StratifiedKFold',
    folds: 5,
    stratified: true,
    shuffle: true,
    random_seed: 42,
    validation_subjects: 'all active training rows',
    synthetic_validation_rows: 500,
  },
}

const globalImportance = [
  ['VAS score', 40],
  ['BMI', 30],
  ['age', 20],
  ['gender', 10],
].map(([feature, relative_importance]) => ({
  feature,
  standardized_coefficient: 0.5,
  direction_when_feature_increases: 'toward_oa',
  relative_importance,
}))

const evaluation = {
  selected_model: 'Logistic Regression',
  validation_strategy: 'Stratified 5-fold cross-validation on the unified active dataset.',
  candidate_models: [
    {
      model: 'Logistic Regression',
      accuracy: 0.71,
      accuracy_std: 0.06,
      precision: 0.70,
      precision_std: 0.08,
      recall: 0.77,
      recall_std: 0.08,
      f1: 0.73,
      f1_std: 0.05,
    },
  ],
  confusion_matrix: {
    labels: ['Healthy', 'Knee OA'],
    matrix: [[30, 15], [10, 33]],
    evaluated_on: 'all active out-of-fold rows',
  },
  global_feature_importance_method: 'Normalized coefficient importance.',
  global_feature_importance: globalImportance,
  disclaimer: 'Internal prototype evaluation, not clinical validation.',
}

const result = {
  prediction: 1,
  result: 'OA Risk Detected',
  risk_level: 'Moderate',
  healthy_probability: 38.246,
  oa_probability: 61.754,
  model_name: 'Logistic Regression',
  model_version: 'oa-model-test',
  feature_values_used: { age: 60, gender: 0, BMI: 25, VAS_score: 5 },
  explanation: {
    local_feature_contributions: [
      ['VAS score', 5, 40, 'toward_oa'],
      ['BMI', 25, 30, 'toward_oa'],
      ['age', 60, 20, 'toward_healthy'],
      ['gender', 0, 10, 'toward_healthy'],
    ].map(([feature, input_value, relative_contribution, contribution_direction]) => ({
      feature,
      input_value,
      relative_contribution,
      contribution_direction,
    })),
    disclaimer: 'This model-based explanation does not establish causes.',
  },
  message: 'Further clinical assessment may be recommended.',
  disclaimer: 'This is an AI-assisted screening prototype and not a medical diagnosis.',
}

before(async () => {
  vite = await createServer({
    root: FRONTEND_ROOT,
    logLevel: 'silent',
    appType: 'custom',
    server: { middlewareMode: true },
  })

  ScreeningForm = (await vite.ssrLoadModule('/src/components/ScreeningForm.jsx')).default
  buildPredictionPayload = (await vite.ssrLoadModule('/src/services/api.js')).buildPredictionPayload
  AnalysisLoader = (await vite.ssrLoadModule('/src/components/AnalysisLoader.jsx')).default
  ModelInfo = (await vite.ssrLoadModule('/src/components/ModelInfo.jsx')).default
  ResultCard = (await vite.ssrLoadModule('/src/components/ResultCard.jsx')).default
})

after(async () => {
  await vite?.close()
})

describe('screening form logic', () => {
  test('validates required fields and clinical input ranges', () => {
    const invalid = { ...completeForm, age: '17', gender: '', heightCm: '0' }
    const errors = validateStep(0, invalid, null)
    assert.deepEqual(Object.keys(errors).sort(), ['age', 'gender', 'heightCm'])
    assert.deepEqual(validateStep(0, completeForm, calculateBMI('165', '68')), {})
    assert.deepEqual(validateStep(1, completeForm, calculateBMI('165', '68')), {})
    assert.deepEqual(validateStep(2, completeForm, calculateBMI('165', '68')), {})
  })

  test('calculates BMI to the same single decimal shown by the form', () => {
    assert.equal(calculateBMI(165, 68), 25)
    assert.equal(calculateBMI('', 68), null)
    assert.equal(calculateBMI(165, 0), null)
  })

  test('advances only when the current step has no validation errors', () => {
    assert.equal(nextStepAfterValidation(0, {}), 1)
    assert.equal(nextStepAfterValidation(1, {}), 2)
    assert.equal(nextStepAfterValidation(2, {}), 2)
    assert.equal(nextStepAfterValidation(0, { age: 'Invalid' }), 0)
  })

  test('renders accessible initial form controls', () => {
    const html = renderToStaticMarkup(React.createElement(ScreeningForm, {
      onSubmit: () => {},
      onReset: () => {},
      isSubmitting: false,
      submitError: '',
    }))
    assert.match(html, /<form[^>]*noValidate=""/)
    assert.match(html, /<label for="age">Age<\/label>/)
    assert.match(html, /AI MODEL INPUTS/)
    assert.match(html, /The current AI model uses age, gender, BMI, and pain score for prediction\./)
    assert.match(html, /BMI is sent to the model\./)
    assert.match(html, /Calculated BMI/)
    assert.match(html, /Screening progress/)
  })

  test('sends only the four current model inputs in the prediction payload', () => {
    const payload = buildPredictionPayload({
      ...completeForm,
      BMI: calculateBMI(completeForm.heightCm, completeForm.weightKg),
    })
    assert.deepEqual(payload, {
      age: '60',
      gender: '0',
      BMI: 25,
      VAS_score: 5,
    })
    assert.deepEqual(Object.keys(payload), ['age', 'gender', 'BMI', 'VAS_score'])
    assert.equal(payload.BMI, 25)
  })
})

describe('API-driven interface states', () => {
  test('renders prediction and model-evidence loading states', () => {
    const predictionLoader = renderToStaticMarkup(React.createElement(AnalysisLoader))
    assert.match(predictionLoader, /aria-busy="true"/)
    assert.match(predictionLoader, /Analyzing OA Risk Markers/)

    const modelLoader = renderToStaticMarkup(React.createElement(ModelInfo, {
      modelInfo: null,
      evaluation: null,
      isLoading: true,
      error: '',
      onRetry: () => {},
    }))
    assert.match(modelLoader, /Loading model evidence/)
    assert.match(modelLoader, /role="status"/)
  })

  test('renders an accessible API error state with retry control', () => {
    const modelErrorHtml = renderToStaticMarkup(React.createElement(ModelInfo, {
      modelInfo: null,
      evaluation: null,
      isLoading: false,
      error: 'Service unavailable',
      onRetry: () => {},
    }))
    assert.match(modelErrorHtml, /role="alert"/)
    assert.match(modelErrorHtml, /Service unavailable/)
    assert.match(modelErrorHtml, /Retry loading/)

    const predictionErrorHtml = renderToStaticMarkup(React.createElement(ScreeningForm, {
      onSubmit: () => {},
      onReset: () => {},
      isSubmitting: false,
      submitError: 'Prediction service unavailable',
    }))
    assert.match(predictionErrorHtml, /role="alert"/)
    assert.match(predictionErrorHtml, /Prediction service unavailable/)
  })

  test('renders the complete prediction result and four feature contributions', () => {
    const html = renderToStaticMarkup(React.createElement(ResultCard, {
      result,
      patient: completeForm,
      modelInfo,
    }))
    assert.match(html, /AI Screening Result/)
    assert.match(html, /61\.8%/)
    assert.match(html, /38\.2%/)
    assert.match(html, /MODERATE RISK/)
    assert.match(html, /Why did the model produce this result\?/)
    assert.equal((html.match(/result-contribution-card/g) || []).length, 4)
    assert.match(html, />40%<\/strong>/)
    assert.doesNotMatch(html, />100%<\/strong>/)
    assert.match(html, /not a medical diagnosis/)
  })

  test('renders backend evaluation metrics, confusion matrix, and importance', () => {
    const html = renderToStaticMarkup(React.createElement(ModelInfo, {
      modelInfo,
      evaluation,
      isLoading: false,
      error: '',
      onRetry: () => {},
    }))
    assert.match(html, /Model comparison/)
    assert.match(html, /73\.0%/)
    assert.match(html, /Confusion matrix/)
    assert.match(html, /Actual Healthy/)
    assert.match(html, /Global feature importance/)
    assert.match(html, /40\.0%/)
  })
})

test('responsive rules cover result and evaluation layouts', async () => {
  const css = await readFile(path.join(FRONTEND_ROOT, 'src', 'App.css'), 'utf8')
  assert.match(css, /@media \(max-width: 980px\)/)
  assert.match(css, /@media \(max-width: 760px\)/)
  assert.match(css, /@media \(max-width: 540px\)/)
  assert.match(css, /\.result-screening-content\s*\{[\s\S]*?grid-template-columns:\s*1fr;/)
  assert.match(css, /\.result-contribution-grid\s*\{[\s\S]*?grid-template-columns:\s*1fr;/)
  assert.match(css, /\.table-scroll\s*\{[\s\S]*?overflow-x:\s*auto;/)
})
