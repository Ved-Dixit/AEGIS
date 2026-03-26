from pathlib import Path

from aegis.data.synthetic import generate_synthetic_supply_chain_dataset
from aegis.service import AegisService, train_model


def test_train_and_predict(tmp_path: Path) -> None:
    frame = generate_synthetic_supply_chain_dataset(rows=1_400, seed=11)
    model_path = tmp_path / "aegis_test_bundle.joblib"

    summary = train_model(data_frame=frame, model_path=model_path)
    assert model_path.exists()
    assert summary["metrics"]["roc_auc"] >= 0.9

    service = AegisService(model_path=model_path)
    record = frame.drop(columns=["is_fraud", "fraud_type"]).iloc[0].to_dict()
    result = service.predict_record(record)

    assert 0.0 <= result["final_score"] <= 1.0
    assert result["risk_band"] in {"low", "medium", "high", "critical"}
    assert result["duplicate_matches"]

