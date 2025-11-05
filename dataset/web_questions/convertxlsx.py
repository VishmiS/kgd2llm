import pandas as pd
import os


# ----------------------------
# Function to convert TSV to XLSX
# ----------------------------
def convert_tsv_to_xlsx(folder_path):
    """
    Converts all TSV files in the given folder (and subfolders) to XLSX files.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".tsv"):
                tsv_path = os.path.join(root, file)
                xlsx_path = os.path.join(root, file.replace(".tsv", ".xlsx"))

                # Read TSV
                df = pd.read_csv(tsv_path, sep="\t")

                # Save as XLSX
                df.to_excel(xlsx_path, index=False)
                print(f"✅ Converted {tsv_path} → {xlsx_path}")


# ----------------------------
# Convert TSV files in all splits
# ----------------------------
splits = ["train"]

for split in splits:
    if os.path.exists(split):
        print(f"\n🔄 Converting TSV files in '{split}' folder...")
        convert_tsv_to_xlsx(split)
    else:
        print(f"⚠️ Folder '{split}' not found!")
