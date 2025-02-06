import pickle

# List of specific files to merge
selected_files = ["train6-1.pkl", "train6-2.pkl", "train6-3.pkl", "train6-4.pkl"]

merged_data = []

# Load each selected file and append its contents
for file in selected_files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
        merged_data.append(data)

# Save merged data into a new .pkl file
with open("train6.pkl", 'wb') as f:
    pickle.dump(merged_data, f)

print("Merging completee")
