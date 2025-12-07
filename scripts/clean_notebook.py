import nbformat

with open(
    r"C:\Users\DELL\OneDrive\Data Science\Bank_Telemarketing_Prediction_EDSB25_1\notebooks\Masterfile_Predicting_The_Success_Of_Bank_Telemarketing_Calls.ipynb",
    encoding="utf-8"
) as f:
    nb = nbformat.read(f, as_version=4)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

with open(
    r"C:\Users\DELL\OneDrive\Data Science\Bank_Telemarketing_Prediction_EDSB25_1\notebooks\Masterfile_cleaned.ipynb",
    "w",
    encoding="utf-8"
) as f:
    nbformat.write(nb, f)

print("Notebook cleaned and saved as Masterfile_cleaned.ipynb")
