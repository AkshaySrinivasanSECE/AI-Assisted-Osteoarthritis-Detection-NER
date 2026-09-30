# Test suite

Run every automated check from the project root:

```powershell
python run_tests.py
```

The runner uses the project `.venv` when it is available and executes:

- Python `unittest` coverage for data integrity, isolated end-to-end training, cross-validation leakage controls, saved artifacts, deterministic predictions, API contracts, request validation, and explainability calculations.
- Node's built-in test runner with Vite SSR for form logic, BMI calculation, step transitions, loading and error states, prediction rendering, evaluation rendering, and responsive-layout contracts.
- Frontend lint and production compilation.

The end-to-end training test writes only to a temporary directory. It does not overwrite the production model, metadata, data, or evaluation artifacts.
