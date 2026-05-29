import pandas as pd
import altair as alt

# 1. Define the raw data structure
# This structure combines the latest T=4, 8, 16, 32 (Total Flow) data with 
# the previously established Small Cache I/O and Peak On-Chip (Threshold) values.
raw_data_long = []
seq_lengths = [4, 8, 16, 32]

# Small Cache I/O and Thresholds (KB) are manually re-introduced from the previous rich dataset.
model_data_list = [
    {'Model': 'Bert', 'Threshold (KB)': 16, 'Small Cache I/O': 132.5, 'data': {
        4: {'Prop': (972.963, 0.463), 'Flash': (530.117, 5.359)},
        8: {'Prop': (494.044, 1.544), 'Flash': (278.602, 11.344)},
        16: {'Prop': (258.059, 5.559), 'Flash': (163.696, 25.188)},
        32: {'Prop': (153.5, 21), 'Flash': (134.508, 60.375)}
    }},
    {'Model': 'GPT-2', 'Threshold (KB)': 16, 'Small Cache I/O': 293.016, 'data': {
        4: {'Prop': (3889.463, 0.463), 'Flash': (2100.875, 5.359)},
        8: {'Prop': (1970.544, 1.544), 'Flash': (1076.86, 11.344)},
        16: {'Prop': (1014.559, 5.559), 'Flash': (575.704, 25.188)},
        32: {'Prop': (550, 21), 'Flash': (353.391, 60.375)}
    }},
    {'Model': 'LLaMA2', 'Threshold (KB)': 64, 'Small Cache I/O': 2196.031, 'data': {
        4: {'Prop': (30916.463, 0.463), 'Flash': (16584.39, 10.359)},
        8: {'Prop': (15557.544, 1.544), 'Flash': (8379.375, 21.344)},
        16: {'Prop': (7881.559, 5.559), 'Flash': (4295.219, 45.188)},
        32: {'Prop': (4057, 21), 'Flash': (2296.406, 100.375)}
    }},
    {'Model': 'Mistral 7B', 'Threshold (KB)': 64, 'Small Cache I/O': 8756.062, 'data': {
        4: {'Prop': (123656.463, 0.463), 'Flash': (66278.421, 10.359)},
        8: {'Prop': (62217.544, 1.544), 'Flash': (33425.406, 21.344)},
        16: {'Prop': (31501.559, 5.559), 'Flash': (17017.25, 45.188)},
        32: {'Prop': (16157, 21), 'Flash': (8856.437, 100.375)}
    }},
    {'Model': 'Phi-3 Mini', 'Threshold (KB)': 36, 'Small Cache I/O': 1680.531, 'data': {
        4: {'Prop': (23235.463, 0.463), 'Flash': (12482.39, 7.859)},
        8: {'Prop': (11716.544, 1.544), 'Flash': (6322.875, 16.344)},
        16: {'Prop': (5960.559, 5.559), 'Flash': (3257.719, 35.188)},
        32: {'Prop': (3096, 21), 'Flash': (1760.906, 80.375)}
    }},
    # XLM and PaLM 8B, Gemma 7B are missing from the latest table
]


# 2. Loop and structure the data into a long-form DataFrame
raw_data_long = []
threshold_data_long = []

for model_entry in model_data_list:
    model_name = model_entry['Model']
    threshold = model_entry['Threshold (KB)']
    
    # Store threshold for line drawing
    threshold_data_long.append({'Model': model_name, 'Threshold (KB)': threshold})
    
    # --- Add "Small Cache" Data Point ---
    small_cache_io = model_entry.get('Small Cache I/O')
    if small_cache_io:
        raw_data_long.append({
            'Model': model_name,
            'T': 'SC', # Label for Small Cache (SC)
            'Type': 'Small Cache',
            'Off-Chip I/O (MB)': small_cache_io,
            'Peak On-Chip (KB)': threshold, # Plotted on the threshold line
            'Threshold (KB)': threshold
        })

    # Add Prop/Flash Data (T=4, 8, 16, 32)
    for T in seq_lengths:
        if T not in model_entry['data']: continue
            
        for type_name, (io, peak) in model_entry['data'][T].items():
            # Data Cleaning (Log scale cannot plot 0 or negative values)
            if io is None or io <= 0: continue
            if peak is None or peak <= 0: continue
                
            raw_data_long.append({
                'Model': model_name,
                'T': T,
                'Type': type_name,
                'Off-Chip I/O (MB)': io,
                'Peak On-Chip (KB)': peak,
                'Threshold (KB)': threshold
            })

df = pd.DataFrame(raw_data_long)
df_thresholds = pd.DataFrame(threshold_data_long)
df['T_str'] = 'T=' + df['T'].astype(str)


# 3. Create Altair chart
# Filter thresholds to only models that appear in the data
models_in_data = df['Model'].unique()
df_thresholds_flow = df_thresholds[df_thresholds['Model'].isin(models_in_data)]

# --- Layer 1: Base Chart Setup ---
base = alt.Chart(df).properties(
    title='Off-Chip I/O vs. Peak On-Chip for Total Flow (Log-Log Scale with Cache Thresholds)',
    width=1000,
    height=700 
)

# --- Layer 2: Scatter Plot (Points) ---
points = base.mark_point(filled=True, size=100).encode(
    # --- MODIFICATION: X-axis domain starts at 10 ---
    x=alt.X('Off-Chip I/O (MB):Q', 
            scale=alt.Scale(type='log', domainMin=10), 
            title='Off-Chip I/O (MB) [Log Scale]'),
    
    y=alt.Y('Peak On-Chip (KB):Q', 
            scale=alt.Scale(type='log'), 
            title='Peak On-Chip (KB) [Log Scale]'),
    
    # Color: Prop(Red), Flash(Blue), Small Cache(Green)
    color=alt.Color('Type:N', 
                    scale=alt.Scale(domain=['Prop', 'Flash', 'Small Cache'], range=['red', 'blue', 'green']), 
                    title='구현 방식'), # Implementation
    shape=alt.Shape('Model:N', title='모델'), # Model
    
    tooltip=['Model', 'T', 'Type', 'Off-Chip I/O (MB)', 'Peak On-Chip (KB)']
)

# --- Layer 3: Text Labels for T ---
t_labels = base.mark_text(align='left', baseline='middle', dx=10).encode(
    x='Off-Chip I/O (MB):Q',
    y='Peak On-Chip (KB):Q',
    text='T_str:N',
    color=alt.value('black')
)

# --- Layer 4: Threshold Lines (Dashed) ---
threshold_lines = alt.Chart(df_thresholds_flow).mark_rule(strokeDash=[5,5]).encode(
    y=alt.Y('Threshold (KB):Q'),
    color=alt.Color('Model:N', legend=None), # Hide distinct legend for lines
    tooltip=['Model', 'Threshold (KB)']
)

# --- Layer 5: Threshold Line Labels (Text) ---
threshold_labels = alt.Chart(df_thresholds_flow).mark_text(
    align='left', 
    baseline='bottom', 
    dx=5, 
    x=5, # Position label 5px from left edge
    dy=-2 # Move text 2px above the line
).encode(
    y='Threshold (KB):Q',
    text=alt.Text('Model:N'),
    color=alt.Color('Model:N', title='캐시 임계값 (KB)') # Cache Threshold (KB)
)

# Combine all layers
chart = (threshold_lines + threshold_labels + points + t_labels).interactive()

# Save
file_name = 'off_chip_io_vs_peak_on_chip_total_flow.png'
chart.save(file_name) 
    
print("---")
print(f"  - {file_name} generated")
print("---")