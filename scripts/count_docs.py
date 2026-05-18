import os

FOLDER = "bct_documents"

circulaires = 0
notes = 0
other = 0

for file in os.listdir(FOLDER):
    name = file.lower()

    if "circulaire" in name:
        circulaires += 1
    elif "note" in name:
        notes += 1
    else:
        other += 1

print(f"Circulaires: {circulaires}")
print(f"Notes: {notes}")
print(f"Other: {other}")
print(f"Total: {circulaires + notes + other}")