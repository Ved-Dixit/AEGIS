"""Streamlit console for exploring AEGIS predictions."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from aegis.config import DATA_DIR, DEFAULT_MODEL_PATH, RAW_DATA_DIR
from aegis.data.sources import describe_sources
from aegis.data.synthetic import generate_synthetic_supply_chain_dataset
from aegis.service import AegisService, train_model

st.set_page_config(
    page_title="AEGIS Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_demo_transactions() -> pd.DataFrame:
    # The UI keeps a small cached sample so users can explore the model instantly.
    return generate_synthetic_supply_chain_dataset(rows=120, seed=7)


st.title("AEGIS Predictive Fraud Intelligence System")
st.caption(
    "Real-time fraud scoring before disbursement, with duplicate detection, graph intelligence, anomaly scoring, and explainability."
)

service = AegisService()
demo_frame = load_demo_transactions()

with st.sidebar:
    st.header("Model Control")
    train_rows = st.slider("Synthetic training rows", min_value=2_000, max_value=12_000, value=6_000, step=1_000)
    available_training_sets = {
        "Synthetic generator": None,
    }
    hybrid_path = DATA_DIR / "dataco_hybrid_aegis.csv"
    if hybrid_path.exists():
        available_training_sets["Hybrid DataCo + injected fraud"] = str(hybrid_path)
    training_choice = st.selectbox("Training dataset", list(available_training_sets.keys()))
    if st.button("Train / Refresh Model", use_container_width=True):
        with st.spinner("Training the AEGIS ensemble..."):
            selected_data_path = available_training_sets[training_choice]
            if selected_data_path:
                training_frame = pd.read_csv(selected_data_path, parse_dates=["invoice_date", "due_date"])
                summary = train_model(
                    rows=len(training_frame),
                    data_frame=training_frame,
                    source_name="dataco_hybrid",
                )
            else:
                summary = train_model(rows=train_rows, source_name="synthetic")
            service._engine = None
        st.success("Model bundle refreshed.")
        st.json(summary)

    st.subheader("Artifact Status")
    st.write(f"Model path: `{DEFAULT_MODEL_PATH}`")
    st.write("Status: ready" if DEFAULT_MODEL_PATH.exists() else "Status: not trained yet")

    st.subheader("Open-source Dataset Catalog")
    st.dataframe(pd.DataFrame(describe_sources()), use_container_width=True, hide_index=True)

sample_transaction_id = st.selectbox("Start from a demo transaction", demo_frame["transaction_id"])
sample_record = demo_frame.loc[demo_frame["transaction_id"] == sample_transaction_id].iloc[0].to_dict()

col_form, col_results = st.columns([1.2, 1.0])

with col_form:
    st.subheader("Transaction Input")
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        transaction_id = col1.text_input("Transaction ID", value=str(sample_record["transaction_id"]))
        invoice_id = col2.text_input("Invoice ID", value=str(sample_record["invoice_id"]))

        col1, col2, col3 = st.columns(3)
        buyer_id = col1.text_input("Buyer ID", value=str(sample_record["buyer_id"]))
        supplier_id = col2.text_input("Supplier ID", value=str(sample_record["supplier_id"]))
        lender_id = col3.text_input("Lender ID", value=str(sample_record["lender_id"]))

        col1, col2, col3 = st.columns(3)
        product_id = col1.text_input("Product ID", value=str(sample_record["product_id"]))
        quantity = col2.number_input("Quantity", min_value=1.0, value=float(sample_record["quantity"]))
        unit_price = col3.number_input("Unit Price", min_value=1.0, value=float(sample_record["unit_price"]))

        invoice_amount_default = float(round(quantity * unit_price, 2))
        col1, col2 = st.columns(2)
        invoice_amount = col1.number_input(
            "Invoice Amount",
            min_value=1.0,
            value=float(sample_record.get("invoice_amount", invoice_amount_default)),
        )
        loan_amount = col2.number_input(
            "Loan Amount",
            min_value=1.0,
            value=float(sample_record["loan_amount"]),
        )

        col1, col2 = st.columns(2)
        invoice_date = col1.date_input(
            "Invoice Date",
            value=pd.Timestamp(sample_record["invoice_date"]).date() if sample_record.get("invoice_date") else date.today(),
        )
        due_date = col2.date_input(
            "Due Date",
            value=pd.Timestamp(sample_record["due_date"]).date()
            if sample_record.get("due_date")
            else date.today() + timedelta(days=30),
        )

        col1, col2, col3 = st.columns(3)
        payment_term_days = col1.number_input(
            "Payment Term Days",
            min_value=1,
            value=int(sample_record["payment_term_days"]),
        )
        shipment_distance_km = col2.number_input(
            "Shipment Distance (km)",
            min_value=0.0,
            value=float(sample_record["shipment_distance_km"]),
        )
        historic_late_payments = col3.number_input(
            "Historic Late Payments",
            min_value=0,
            value=int(sample_record["historic_late_payments"]),
        )

        col1, col2, col3, col4 = st.columns(4)
        buyer_risk_rating = col1.slider("Buyer Risk", min_value=0.0, max_value=1.0, value=float(sample_record["buyer_risk_rating"]))
        supplier_risk_rating = col2.slider(
            "Supplier Risk",
            min_value=0.0,
            max_value=1.0,
            value=float(sample_record["supplier_risk_rating"]),
        )
        prior_financing_count = col3.number_input(
            "Prior Financing Count",
            min_value=0,
            value=int(sample_record["prior_financing_count"]),
        )
        channel = col4.selectbox("Channel", options=["portal", "edi", "manual"], index=["portal", "edi", "manual"].index(str(sample_record["channel"])))

        currency = st.selectbox("Currency", options=["USD", "EUR", "INR"], index=["USD", "EUR", "INR"].index(str(sample_record["currency"])))
        submitted = st.form_submit_button("Score Transaction", use_container_width=True)

if submitted:
    if not DEFAULT_MODEL_PATH.exists():
        with st.spinner("No model bundle found, training a fresh one first..."):
            train_model(rows=6_000)
            service._engine = None

    request_record = {
        "transaction_id": transaction_id,
        "invoice_id": invoice_id,
        "buyer_id": buyer_id,
        "supplier_id": supplier_id,
        "lender_id": lender_id,
        "product_id": product_id,
        "quantity": quantity,
        "unit_price": unit_price,
        "invoice_amount": invoice_amount,
        "loan_amount": loan_amount,
        "invoice_date": invoice_date,
        "due_date": due_date,
        "payment_term_days": payment_term_days,
        "shipment_distance_km": shipment_distance_km,
        "buyer_risk_rating": buyer_risk_rating,
        "supplier_risk_rating": supplier_risk_rating,
        "historic_late_payments": historic_late_payments,
        "prior_financing_count": prior_financing_count,
        "channel": channel,
        "currency": currency,
    }
    result = service.predict_record(request_record)

    with col_results:
        st.subheader("Decision Layer")
        st.metric("Final Fraud Score", f"{result['final_score']:.2%}")
        st.metric("Risk Band", result["risk_band"].upper())
        st.progress(float(result["final_score"]))

        component_frame = pd.DataFrame(
            [
                {"component": "classifier", "score": result["classifier_score"]},
                {"component": "anomaly", "score": result["anomaly_score"]},
                {"component": "duplicate", "score": result["duplicate_score"]},
                {"component": "graph", "score": result["graph_score"]},
            ]
        ).set_index("component")
        st.bar_chart(component_frame)

        st.subheader("Explainable AI")
        for explanation in result["explanations"]:
            st.write(f"- {explanation}")

        st.subheader("Nearest Duplicate Matches")
        duplicate_matches = pd.DataFrame(result["duplicate_matches"])
        if duplicate_matches.empty:
            st.write("No close duplicate neighbours were found.")
        else:
            st.dataframe(duplicate_matches, use_container_width=True, hide_index=True)

else:
    with col_results:
        st.subheader("Decision Layer")
        st.write("Submit a transaction to get a full fraud breakdown.")
        st.code(
            "uvicorn backend.main:app --reload\nstreamlit run streamlit_app.py",
            language="bash",
        )

st.divider()
history_tab, datasets_tab = st.tabs(["History", "External Datasets"])

with history_tab:
    history_col1, history_col2, history_col3 = st.columns(3)

    with history_col1:
        st.subheader("Training Runs")
        training_history = pd.DataFrame(service.recent_training_runs(limit=10))
        if training_history.empty:
            st.write("No training runs recorded yet.")
        else:
            st.dataframe(training_history[["created_at", "source_name", "rows", "fraud_rows"]], hide_index=True)

    with history_col2:
        st.subheader("Dataset Imports")
        dataset_history = pd.DataFrame(service.recent_dataset_imports(limit=10))
        if dataset_history.empty:
            st.write("No dataset imports recorded yet.")
        else:
            st.dataframe(dataset_history[["created_at", "source_name", "row_count", "output_path"]], hide_index=True)

    with history_col3:
        st.subheader("Recent Predictions")
        prediction_history = pd.DataFrame(service.recent_predictions(limit=15))
        if prediction_history.empty:
            st.write("No scored transactions recorded yet.")
        else:
            st.dataframe(
                prediction_history[["created_at", "transaction_id", "invoice_id", "final_score", "risk_band"]],
                hide_index=True,
            )

with datasets_tab:
    st.subheader("Prepare Kaggle / GitHub Data")
    connector_catalog = pd.DataFrame(service.public_source_catalog())
    selected_source = st.selectbox("Source preset", connector_catalog["key"])
    selected_meta = connector_catalog.loc[connector_catalog["key"] == selected_source].iloc[0].to_dict()
    st.json(selected_meta)

    controls_col1, controls_col2 = st.columns(2)
    with controls_col1:
        if st.button("Fetch Source Files / Instructions", use_container_width=True):
            fetch_summary = service.fetch_public_source(selected_source, execute_kaggle=False)
            st.json(fetch_summary)

    with controls_col2:
        if st.button("Download and Prepare If Directly Available", use_container_width=True):
            fetch_summary = service.fetch_public_source(selected_source, execute_kaggle=False)
            if fetch_summary["status"] == "downloaded":
                prepared = service.prepare_external_dataset(
                    source_name=selected_source,
                    input_paths=fetch_summary["files"],
                    output_path=str(DATA_DIR / f"{selected_source}_aegis.csv"),
                )
                st.success("Dataset prepared.")
                st.json(prepared)
                st.dataframe(pd.read_csv(prepared["output_path"]).head(15), use_container_width=True, hide_index=True)
            else:
                st.warning(fetch_summary["notes"])
                if "commands" in fetch_summary:
                    st.code("\n".join(fetch_summary["commands"]), language="bash")

    uploaded_files = st.file_uploader(
        "Or upload raw CSV file(s) and normalize them into the AEGIS schema",
        type=["csv"],
        accept_multiple_files=True,
    )
    if st.button("Prepare Uploaded Dataset", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload the raw CSV file(s) first.")
        else:
            upload_dir = RAW_DATA_DIR / "uploads" / selected_source
            upload_dir.mkdir(parents=True, exist_ok=True)
            input_paths: list[str] = []
            for uploaded_file in uploaded_files:
                destination = upload_dir / Path(uploaded_file.name).name
                destination.write_bytes(uploaded_file.getvalue())
                input_paths.append(str(destination))

            prepared = service.prepare_external_dataset(
                source_name=selected_source,
                input_paths=input_paths,
                output_path=str(DATA_DIR / f"{selected_source}_uploaded_aegis.csv"),
            )
            st.success("Uploaded dataset prepared.")
            st.json(prepared)
            st.dataframe(pd.read_csv(prepared["output_path"]).head(15), use_container_width=True, hide_index=True)
