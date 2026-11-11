
import json
import pandas as pd
file_path = "models/qlearning/q_table_20261104_151256.json"

with open(file_path, "r") as file:
    data = json.load(file)

# Extract the Q-table
q_table = data.get("q_table", data)

rows = []

for state, actions in q_table.items():
    if isinstance(actions, dict):
        for action, value in actions.items():
            if isinstance(value, (int, float)):
                rows.append({
                    "State": state,
                    "Action": action,
                    "Q-value": round(value, 4)
                })

df = pd.DataFrame(rows)

print(df.to_string(index=False))

# Save the table as a CSV file
df.to_csv("results/training/qtable_view.csv", index=False)

print("\nQ-table saved to results/training/qtable_view.csv")
