import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# -----------------------------
# Step 1: Load dataset
# -----------------------------
data = pd.read_csv('container_data.csv')

# -----------------------------
# Step 2: Preprocess data
# -----------------------------
selected_features = [
    'ContainerID', 'CargoTemp', 'Setpoint', 'MinSetpoint', 'MaxSetpoint',
    'IsPerishable', 'CargoType', 'CargoDescription', 'PowerStatus', 'AlarmsPresent',
    'Power', 'CumulativePower', 'ControlModeCompressorState', 'OperatingModeName'
]

df = data[selected_features].copy()

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Encode categorical features for clustering
categorical_cols = ['PowerStatus', 'AlarmsPresent', 'ControlModeCompressorState', 'OperatingModeName']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_imputed[col] = le.fit_transform(df_imputed[col])
    label_encoders[col] = le

# Normalize numeric features for clustering
numeric_cols = ['CargoTemp', 'Setpoint', 'MinSetpoint', 'MaxSetpoint', 'Power', 'CumulativePower']
scaler = StandardScaler()
df_imputed[numeric_cols] = scaler.fit_transform(df_imputed[numeric_cols])

# -----------------------------
# Step 3: Apply clustering (optional for grouping)
# -----------------------------
X = df_imputed.drop(columns=['ContainerID', 'CargoType', 'CargoDescription', 'IsPerishable'])
kmeans = KMeans(n_clusters=3, random_state=42)
df_imputed['Cluster'] = kmeans.fit_predict(X)

# -----------------------------
# Step 4: Enhanced perishable logic
# -----------------------------
def is_perishable(row):
    if str(row['IsPerishable']).lower() in ['true', '1']:
        return True
    perish_keywords = ['fruit', 'vegetable', 'meat', 'dairy', 'fish', 'fresh']
    if any(kw in str(row['CargoType']).lower() for kw in perish_keywords) or \
       any(kw in str(row['CargoDescription']).lower() for kw in perish_keywords):
        return True
    try:
        sp = float(row['Setpoint'])
        if sp <= -10 or (0 <= sp <= 13):
            return True
    except:
        pass
    return False

def within_tolerance(row):
    try:
        return (float(row['CargoTemp']) >= float(row['MinSetpoint'])) and \
               (float(row['CargoTemp']) <= float(row['MaxSetpoint']))
    except:
        return False

# -----------------------------
# Step 5: Container-level outputs
# -----------------------------
safety_margin = 1.2
duration_hours = 3

results = []
for idx, row in df.iterrows():
    perishable = is_perishable(row)
    alarms = str(row['AlarmsPresent']).lower() in ['true', '1']
    safe_mode = str(row['ControlModeCompressorState']).lower() in ['off', '0']
    can_power_off = ((not perishable) or within_tolerance(row)) and (not alarms) and safe_mode
    try:
        power_w = float(row['Power'])
    except:
        power_w = 0.0
    power_required_kwh = (power_w * duration_hours * safety_margin) / 1000.0
    results.append({
        'ContainerID': row['ContainerID'],
        'CanPowerOff': can_power_off,
        'PowerRequired_3hrs_kWh': round(power_required_kwh, 2)
    })

# Convert to DataFrame for easy export
output_df = pd.DataFrame(results)

# -----------------------------
# Step 6: Display results
# -----------------------------
print(output_df)

# Optional: Save to CSV
output_df.to_csv('container_power_analysis.csv', index=False)