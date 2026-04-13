import uuid
import numpy as np
import pandas as pd
import gradio as gr

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED

import joblib
from sklearn.neighbors import NearestNeighbors

MODEL_PATH = "cox2_rf_model.pkl"
TRAIN_FPS_PATH = "train_fps.npy"

FP_NBITS = 2048
FP_RADIUS = 2
KNN_K = 5
AD_PERCENTILE = 5
THRESHOLD = 0.5


def smiles_to_fp_bits(smiles: str, n_bits: int = FP_NBITS, radius: int = FP_RADIUS) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calc_mol_properties(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string for property calculation.")

    mol_wt = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    qed_score = float(QED.qed(mol))

    try:
        num_rings = mol.GetRingInfo().NumRings()
    except Exception:
        num_rings = 0

    sa_score = (mol_wt / 500.0) + (rot_bonds / 10.0) + (num_rings / 5.0)

    return {
        "MolWt": round(mol_wt, 2),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "HBD": int(hbd),
        "HBA": int(hba),
        "RotatableBonds": int(rot_bonds),
        "QED": round(qed_score, 3),
        "SA_score": round(sa_score, 3),
    }


def calc_lipinski(props: dict):
    violations = 0
    if props["MolWt"] > 500:
        violations += 1
    if props["LogP"] > 5:
        violations += 1
    if props["HBD"] > 5:
        violations += 1
    if props["HBA"] > 10:
        violations += 1

    lipinski_pass = "Pass" if violations == 0 else "Fail"
    return lipinski_pass, violations


def build_lipinski_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<div style='color:#666;'>No results to display.</div>"

    cols = [c for c in ["SMILES", "Prediction", "Probability (Active)", "Lipinski", "Violations", "In_AD"] if c in df.columns]
    if not cols:
        return "<div style='color:#666;'>No summary columns found.</div>"

    rows_html = []
    for _, r in df[cols].iterrows():
        lip = str(r.get("Lipinski", ""))
        lip_color = "#2e7d32" if lip == "Pass" else ("#c62828" if lip == "Fail" else "#444")

        inad = str(r.get("In_AD", ""))
        ad_color = "#2e7d32" if inad == "Yes" else ("#c62828" if inad == "No" else "#444")

        tds = []
        for c in cols:
            if c == "Lipinski":
                tds.append(
                    f"<td style='padding:6px 8px;border-bottom:1px solid #eee;font-weight:600;color:{lip_color};'>{lip}</td>"
                )
            elif c == "In_AD":
                tds.append(
                    f"<td style='padding:6px 8px;border-bottom:1px solid #eee;font-weight:600;color:{ad_color};'>{inad}</td>"
                )
            else:
                tds.append(f"<td style='padding:6px 8px;border-bottom:1px solid #eee;'>{r.get(c, '')}</td>")

        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    header_html = "".join([f"<th style='text-align:left;padding:6px 8px;border-bottom:2px solid #ddd;'>{c}</th>" for c in cols])
    table_html = (
        "<div style='max-height:320px;overflow:auto;border:1px solid #eee;border-radius:8px;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )
    return table_html


model = joblib.load(MODEL_PATH)
print("MODEL classes_ =", getattr(model, "classes_", None))

AD_ENABLED = False
ad_threshold = None
nn = None
train_fps = None


def _prepare_ad():
    global AD_ENABLED, ad_threshold, nn, train_fps

    try:
        train_fps = np.load(TRAIN_FPS_PATH)
        train_fps = (train_fps > 0).astype(np.int8)

        nn = NearestNeighbors(
            n_neighbors=KNN_K + 1,
            metric="jaccard",
            algorithm="brute",
            n_jobs=-1
        )
        nn.fit(train_fps)

        dists, _ = nn.kneighbors(train_fps, return_distance=True)
        mean_dist = dists[:, 1:].mean(axis=1)
        sims = 1.0 - mean_dist
        ad_threshold = float(np.percentile(sims, AD_PERCENTILE))

        AD_ENABLED = True
        print(f"AD loaded successfully. Threshold = {ad_threshold:.3f}")
    except Exception as e:
        print("AD init failed:", e)
        AD_ENABLED = False


_prepare_ad()


def compute_ad_similarity(fp_bits_1d: np.ndarray):
    if not AD_ENABLED:
        return None, None, "AD not available (train_fps.npy not loaded)."

    x = (fp_bits_1d.reshape(1, -1) > 0).astype(np.int8)
    dists, _ = nn.kneighbors(x, n_neighbors=KNN_K, return_distance=True)
    mean_dist = float(dists.mean())
    sim = 1.0 - mean_dist
    in_ad = sim >= ad_threshold

    warn = "" if in_ad else "Outside AD: prediction should be interpreted with caution."
    return round(sim, 3), ("Yes" if in_ad else "No"), warn


def get_active_probability(model, x: np.ndarray) -> float:
    proba = model.predict_proba(x)[0]
    classes = list(getattr(model, "classes_", []))

    if 1 in classes:
        active_idx = classes.index(1)
        return float(proba[active_idx])

    return float(np.max(proba))


def predict_activity_batch(smiles_text: str, csv_file):
    smiles_list = []

    if smiles_text:
        smiles_list.extend([s.strip() for s in smiles_text.splitlines() if s.strip()])

    if csv_file is not None:
        try:
            df_csv = pd.read_csv(csv_file)
            if "SMILES" in df_csv.columns:
                smiles_list.extend([str(s).strip() for s in df_csv["SMILES"].tolist() if str(s).strip()])
        except Exception as e:
            print("CSV read failed:", e)

    seen, unique_smiles = set(), []
    for s in smiles_list:
        if s not in seen:
            seen.add(s)
            unique_smiles.append(s)

    columns = [
        "SMILES",
        "Prediction",
        "Probability (Active)",
        "AD_similarity",
        "In_AD",
        "AD_warning",
        "MolWt",
        "LogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotatableBonds",
        "QED",
        "SA_score",
        "Lipinski",
        "Violations",
    ]

    rows = []
    for smi in unique_smiles:
        try:
            fp_bits = smiles_to_fp_bits(smi)
            x = fp_bits.reshape(1, -1).astype(np.float32)

            proba_active = get_active_probability(model, x)
            label = "Active" if proba_active >= THRESHOLD else "Inactive"

            ad_sim, in_ad, ad_warn = compute_ad_similarity(fp_bits)
            props = calc_mol_properties(smi)
            lip_pass, lip_viol = calc_lipinski(props)

            rows.append(
                {
                    "SMILES": smi,
                    "Prediction": label,
                    "Probability (Active)": round(proba_active, 3),
                    "AD_similarity": ad_sim,
                    "In_AD": in_ad,
                    "AD_warning": ad_warn,
                    "MolWt": props["MolWt"],
                    "LogP": props["LogP"],
                    "TPSA": props["TPSA"],
                    "HBD": props["HBD"],
                    "HBA": props["HBA"],
                    "RotatableBonds": props["RotatableBonds"],
                    "QED": props["QED"],
                    "SA_score": props["SA_score"],
                    "Lipinski": lip_pass,
                    "Violations": lip_viol,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "SMILES": smi,
                    "Prediction": f"Error: {e}",
                    "Probability (Active)": None,
                    "AD_similarity": None,
                    "In_AD": None,
                    "AD_warning": None,
                    "MolWt": None,
                    "LogP": None,
                    "TPSA": None,
                    "HBD": None,
                    "HBA": None,
                    "RotatableBonds": None,
                    "QED": None,
                    "SA_score": None,
                    "Lipinski": None,
                    "Violations": None,
                }
            )

    df = pd.DataFrame(rows, columns=columns)

    csv_name = f"cox2_predictions_{uuid.uuid4().hex}.csv"
    df.to_csv(csv_name, index=False)

    lip_html = build_lipinski_html(df)
    return df, csv_name, lip_html


EMPTY_COLUMNS = [
    "SMILES",
    "Prediction",
    "Probability (Active)",
    "AD_similarity",
    "In_AD",
    "AD_warning",
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "QED",
    "SA_score",
    "Lipinski",
    "Violations",
]
EMPTY_DF = pd.DataFrame(columns=EMPTY_COLUMNS)


def clear_all():
    return "", None, EMPTY_DF, None, ""


with gr.Blocks(title="COX-2 Activity Predictor") as demo:
    gr.Markdown("# COX-2 Activity Predictor (Random Forest)")
    gr.Markdown(
        "Enter one or more **SMILES** strings (one per line), or upload a CSV file with a `SMILES` column. "
        "The model predicts whether each compound is **Active** or **Inactive** toward COX-2 using a "
        "Random Forest classifier trained on **Morgan fingerprints (radius = 2, 2048 bits)**.\n\n"
        "**Applicability Domain (AD):** kNN (k = 5) with **Jaccard distance** on training fingerprints. "
        f"Threshold is the **{AD_PERCENTILE}th percentile** of the training similarity distribution."
        + (
            f"\n\n✅ AD loaded. Threshold = **{ad_threshold:.3f}**"
            if AD_ENABLED
            else "\n\n⚠️ AD not loaded (train_fps.npy missing or failed to read)."
        )
    )

    with gr.Row():
        smiles_in = gr.Textbox(
            label="SMILES Input (one per line)",
            lines=8,
            placeholder="Enter one SMILES per line"
        )

    with gr.Row():
        csv_in = gr.File(
            label="Optional CSV input (must contain a 'SMILES' column)",
            file_types=[".csv"],
            type="filepath",
        )

    with gr.Row():
        predict_btn = gr.Button("Predict", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    table_out = gr.Dataframe(
        headers=EMPTY_COLUMNS,
        value=EMPTY_DF,
        label="Prediction Results",
        interactive=False,
    )

    file_out = gr.File(
        label="Download results (CSV)",
        type="filepath",
    )

    lipinski_html = gr.HTML(label="Summary (Lipinski + AD)")

    predict_btn.click(
        fn=predict_activity_batch,
        inputs=[smiles_in, csv_in],
        outputs=[table_out, file_out, lipinski_html],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[smiles_in, csv_in, table_out, file_out, lipinski_html],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
