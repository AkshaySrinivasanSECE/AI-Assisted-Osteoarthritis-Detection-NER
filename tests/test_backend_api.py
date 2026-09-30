import asyncio
import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

import backend.main as api


def asgi_request(method, path, payload=None, raw_body=None):
    if raw_body is None:
        raw_body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    headers = [(b"host", b"testserver")]
    if payload is not None or raw_body:
        headers.append((b"content-type", b"application/json"))

    messages = []
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": raw_body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    asyncio.run(api.app(scope, receive, send))
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return start["status"], json.loads(body.decode("utf-8")) if body else None


class BackendContractTests(unittest.TestCase):
    def test_health_reports_loaded_model(self):
        response = api.HealthResponse.model_validate(api.health())
        self.assertEqual(response.api_status, "healthy")
        self.assertTrue(response.model_loaded)

    def test_model_info_contract(self):
        response = api.ModelInfoResponse.model_validate(api.model_info())
        self.assertEqual(response.selected_model, "Logistic Regression")
        self.assertEqual(response.features, ["age", "gender", "BMI", "VAS score"])
        self.assertEqual(response.original_subject_count, 88)
        self.assertEqual(response.synthetic_training_count, 500)
        self.assertEqual(response.total_training_rows, 588)
        self.assertEqual(response.validation_strategy.folds, 5)
        self.assertEqual(response.validation_strategy.synthetic_validation_rows, 0)
        self.assertTrue(response.model_version.startswith("oa-model-"))
        self.assertIsNone(response.training_timestamp)

    def test_evaluation_contract_contains_aggregate_results_only(self):
        raw_response = api.evaluation()
        response = api.EvaluationResponse.model_validate(raw_response)
        self.assertEqual(len(response.candidate_models), 5)
        self.assertEqual(response.selected_model, "Logistic Regression")
        self.assertEqual(response.confusion_matrix.matrix, [[30, 15], [10, 33]])
        self.assertEqual(len(response.global_feature_importance), 4)
        serialized = json.dumps(raw_response).lower()
        self.assertNotIn("subject id", serialized)
        self.assertNotIn("record_id", serialized)
        self.assertNotIn("oof_predictions", serialized)

    def test_predict_contract_contains_required_backend_fields(self):
        raw_response = api.predict(
            api.PatientData(age=60, gender=0, BMI=26.71, VAS_score=5)
        )
        response = api.PredictionResponse.model_validate(raw_response)
        self.assertEqual(response.model_name, "Logistic Regression")
        self.assertTrue(response.model_version.startswith("oa-model-"))
        self.assertEqual(response.feature_values_used.age, 60)
        self.assertEqual(response.feature_values_used.VAS_score, 5)
        self.assertEqual(len(response.explanation.local_feature_contributions), 4)

    def test_request_validation_rejects_extra_fields(self):
        with self.assertRaises(ValidationError):
            api.PatientData.model_validate({
                "age": 60,
                "gender": 0,
                "BMI": 25,
                "VAS_score": 5,
                "patient_name": "not accepted",
            })

    def test_request_validation_rejects_invalid_values(self):
        invalid_payloads = [
            {"age": 17, "gender": 0, "BMI": 25, "VAS_score": 5},
            {"age": 60.5, "gender": 0, "BMI": 25, "VAS_score": 5},
            {"age": 60, "gender": 2, "BMI": 25, "VAS_score": 5},
            {"age": 60, "gender": 0, "BMI": 0, "VAS_score": 5},
            {"age": 60, "gender": 0, "BMI": float("nan"), "VAS_score": 5},
            {"age": 60, "gender": 0, "BMI": 25, "VAS_score": 11},
            {"age": 60, "gender": 0, "BMI": 25, "VAS_score": 5.5},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                api.PatientData.model_validate(payload)

    def test_unavailable_model_returns_controlled_503(self):
        with patch.object(api, "model", None), patch.object(api, "explainer", None):
            degraded = api.HealthResponse.model_validate(api.health())
            self.assertEqual(degraded.api_status, "degraded")
            self.assertFalse(degraded.model_loaded)
            with self.assertRaises(HTTPException) as context:
                api.predict(api.PatientData(age=60, gender=0, BMI=25, VAS_score=5))
            self.assertEqual(context.exception.status_code, 503)

    def test_cors_origins_are_limited_to_local_frontend_development(self):
        self.assertEqual(
            set(api.DEVELOPMENT_ORIGINS),
            {
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:5174",
                "http://127.0.0.1:5174",
            },
        )


class BackendHttpEndpointTests(unittest.TestCase):
    valid_payload = {"age": 60, "gender": 0, "BMI": 26.71, "VAS_score": 5}

    def test_health_endpoint(self):
        status, body = asgi_request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"api_status": "healthy", "model_loaded": True})

    def test_model_info_endpoint(self):
        status, body = asgi_request("GET", "/model-info")
        self.assertEqual(status, 200)
        self.assertEqual(body["original_subject_count"], 88)
        self.assertEqual(body["synthetic_training_count"], 500)
        self.assertEqual(body["validation_strategy"]["synthetic_validation_rows"], 0)

    def test_evaluation_endpoint(self):
        status, body = asgi_request("GET", "/evaluation")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["candidate_models"]), 5)
        self.assertEqual(len(body["global_feature_importance"]), 4)
        self.assertEqual(len(body["confusion_matrix"]["matrix"]), 2)

    def test_predict_endpoint(self):
        status, body = asgi_request("POST", "/predict", self.valid_payload)
        self.assertEqual(status, 200)
        self.assertIn(body["prediction"], [0, 1])
        self.assertAlmostEqual(
            body["oa_probability"] + body["healthy_probability"],
            100.0,
            places=1,
        )
        self.assertEqual(len(body["explanation"]["local_feature_contributions"]), 4)

    def test_invalid_age_is_rejected(self):
        status, body = asgi_request(
            "POST", "/predict", {**self.valid_payload, "age": 17}
        )
        self.assertEqual(status, 422)
        self.assertIn("age", json.dumps(body))

    def test_invalid_bmi_is_rejected(self):
        for invalid_bmi in (0, -1, 81):
            with self.subTest(BMI=invalid_bmi):
                status, body = asgi_request(
                    "POST", "/predict", {**self.valid_payload, "BMI": invalid_bmi}
                )
                self.assertEqual(status, 422)
                self.assertIn("BMI", json.dumps(body))

    def test_invalid_gender_is_rejected(self):
        status, body = asgi_request(
            "POST", "/predict", {**self.valid_payload, "gender": 2}
        )
        self.assertEqual(status, 422)
        self.assertIn("gender", json.dumps(body))

    def test_invalid_vas_score_is_rejected(self):
        for invalid_vas in (-1, 11, 5.5):
            with self.subTest(VAS_score=invalid_vas):
                status, body = asgi_request(
                    "POST",
                    "/predict",
                    {**self.valid_payload, "VAS_score": invalid_vas},
                )
                self.assertEqual(status, 422)
                self.assertIn("VAS_score", json.dumps(body))

    def test_malformed_json_is_rejected(self):
        status, body = asgi_request("POST", "/predict", raw_body=b'{"age":')
        self.assertEqual(status, 422)
        self.assertEqual(body["detail"][0]["type"], "json_invalid")

    def test_missing_fields_are_rejected(self):
        status, body = asgi_request("POST", "/predict", {"age": 60})
        self.assertEqual(status, 422)
        locations = {tuple(item["loc"]) for item in body["detail"]}
        self.assertTrue({
            ("body", "gender"),
            ("body", "BMI"),
            ("body", "VAS_score"),
        }.issubset(locations))


if __name__ == "__main__":
    unittest.main()
