import numpy as np
import json

# ----------------------------
# Configuration
# ----------------------------
INPUT_NPY = "data/demo_pca_chr1.npy"           # original large file
OUTPUT_NPZ = "data/demo_pca_chr1_smaller.npz" # smaller output
TOP_FEATURES = 30                              # only keep top 30 features per PC

# ----------------------------
# Load original data
# ----------------------------
data = np.load(INPUT_NPY, allow_pickle=True).item()

pcs = data["pcs"].astype("float16")           # reduce precision
superpops = data["superpops"]
samples = data.get("samples", np.arange(pcs.shape[0]))

# Keep only the top N features in the hover_texts
with open("data/hover-interp.json", "r") as f:
    hover_texts = json.load(f)

small_hover = {}
for sample, sample_data in hover_texts.items():
    new_sample_data = {}
    new_sample_data["top_pcs"] = sample_data["top_pcs"]
    for pc_rank in sample_data.get("top_pcs", []):
        pc_key = str(pc_rank)
        if pc_key in sample_data:
            pc_block = sample_data[pc_key].copy()
            pc_block["features"] = pc_block["features"][:TOP_FEATURES]
            new_sample_data[pc_key] = pc_block
    small_hover[sample] = new_sample_data

# ----------------------------
# Save smaller .npz file
# ----------------------------
np.savez_compressed(
    OUTPUT_NPZ,
    pcs=pcs,
    superpops=superpops,
    samples=samples,
    hover_texts=small_hover
)

print(f"Saved smaller file: {OUTPUT_NPZ}")
print(f"Original pcs dtype: {data['pcs'].dtype}, New pcs dtype: {pcs.dtype}")
print(f"Top features per PC limited to {TOP_FEATURES}")
