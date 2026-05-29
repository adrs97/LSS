import pandas as pd
import io
import re

# Energy cost parameters (assuming nJ as per comments)
MAC_COST_PER_FLOP = 0.00023 # nJ/op
DRAM_COST_PER_BIT = 0.0008  # nJ/bit
BITS_PER_MB = 1024 * 1024 * 8

# --- Provided Data Table ---
# This is the 15-row data from your new table
data_blob = """
328 168 88 48 132.508 68.508 36.508 20.508 0.463 1.544 5.559 21 3.297 6.969 15.438 36.875 358326000 313078000 291238000 280514000 325619000 296727000 286641000 276426000
644.5 324.5 164.5 84.5 392.25 198.75 102 53.625 0.328 1.281 5.062 20.375 5.359 11.344 25.188 60.375 640667000 590257000 565052000 552450000 776118000 725709000 980728000 687902000
972.5 492.5 252.5 132.5 524.758 267.258 138.508 74.133 0.463 1.544 5.559 21 5.359 11.344 25.188 60.375 999074000 903368000 856302000 832965000 1101590000 1022270000 1266880000 964156000
1312 672 352 192 529.016 273.016 145.016 81.016 0.463 1.544 5.559 21 3.297 6.969 15.438 36.875 1433300000 1252310000 1164950000 1122060000 1302520000 1186910000 1132210000 1105700000
2577 1297 657 337 1566.5 792.5 405.5 212 0.328 1.281 5.062 20.375 5.359 11.344 25.188 60.375 2562370000 2360750000 2259940000 2209530000 3104160000 2902570000 2801730000 2751330000
3889 1969 1009 529 2095.516 1065.516 550.516 293.016 0.463 1.544 5.559 21 5.359 11.344 25.188 60.375 3996000000 3613190000 3424940000 3331590000 4406060000 4088800000 3933310000 3856360000
10368 5248 2688 1408 4164.031 2116.031 1092.031 580.031 0.463 1.544 5.559 21 6.297 12.969 27.438 60.875 11424600000 9980820000 9284040000 8941930000 10366200000 9451330000 9018970000 8809050000
20548 10308 5188 2628 12410 6242 3158 1616 0.328 1.281 5.062 20.375 10.359 21.344 45.188 100.375 20464200000 18851300000 18044800000 17641600000 24781800000 23168100000 22362400000 21959300000
30916 15556 7876 4036 16574.031 8358.031 4250.031 2196.031 0.463 1.544 5.559 21 10.359 21.344 45.188 100.375 31891300000 28833100000 27329200000 26583500000 35142400000 32614900000 31376300000 30763300000
41472 20992 10752 5632 16648.062 8456.062 4360.062 2312.062 0.463 1.544 5.559 21 6.297 12.969 27.438 60.875 45698300000 39923300000 37136200000 35767700000 41462100000 37806500000 36074200000 35233500000
82184 41224 20744 10504 49620 24948 12612 6444 0.328 1.281 5.062 20.375 10.359 21.344 45.188 100.375 81854400000 75402900000 72177100000 70564200000 99126100000 92680400000 89463900000 87852600000
123656 62216 31496 16136 66268.062 33404.062 16972.062 8756.062 0.463 1.544 5.559 21 10.359 21.344 45.188 100.375 127563000000 115330000000 109314000000 106332000000 140567000000 130457000000 125503000000 123052000000
7808 3968 2048 1088 3139.031 1603.031 835.031 451.031 0.463 1.544 5.559 21 4.797 9.969 21.438 48.875 8578900000 7495030000 6971930000 6715080000 7787700000 7099180000 6774020000 6615670000
15427 7747 3907 1987 9335.5 4703.5 2387.5 1229.5 0.328 1.281 5.062 20.375 7.859 16.344 35.188 80.375 15356500000 14146900000 13542000000 13239600000 18598900000 17389300000 16784500000 16481900000
23235 11715 5955 3075 12474.531 6306.531 3222.531 1680.531 0.463 1.544 5.559 21 7.859 16.344 35.188 80.375 23937400000 21642700000 20514200000 19954700000 26383000000 24484700000 23554500000 23094100000
"""

row_labels = [
    ("Bert", "FWD"), ("Bert", "BWD"), ("Bert", "Total"),
    ("GPT-2", "FWD"), ("GPT-2", "BWD"), ("GPT-2", "Total"),
    ("LLaMA2", "FWD"), ("LLaMA2", "BWD"), ("LLaMA2", "Total"),
    ("Mistral 7B", "FWD"), ("Mistral 7B", "BWD"), ("Mistral 7B", "Total"),
    ("Phi-3 Mini", "FWD"), ("Phi-3 Mini", "BWD"), ("Phi-3 Mini", "Total"),
]

# Use io.StringIO to treat the string block as a file
f = io.StringIO(data_blob.strip())

# Define the 24 columns *as they appear in the new data blob*
# This is a "blocked" layout (Metric -> Method -> Tile)
original_columns_map = {
    # Off-Chip I/O (MB) -> Proposed
    0: "IO_Prop_T4", 1: "IO_Prop_T8", 2: "IO_Prop_T16", 3: "IO_Prop_T32",
    # Off-Chip I/O (MB) -> Flash
    4: "IO_Flash_T4", 5: "IO_Flash_T8", 6: "IO_Flash_T16", 7: "IO_Flash_T32",
    
    # Peak On-Chip (KB) -> Proposed
    8: "Peak_Prop_T4", 9: "Peak_Prop_T8", 10: "Peak_Prop_T16", 11: "Peak_Prop_T32",
    # Peak On-Chip (KB) -> Flash
    12: "Peak_Flash_T4", 13: "Peak_Flash_T8", 14: "Peak_Flash_T16", 15: "Peak_Flash_T32",

    # FLOPs -> Proposed
    16: "FLOPs_Prop_T4", 17: "FLOPs_Prop_T8", 18: "FLOPs_Prop_T16", 19: "FLOPs_Prop_T32",
    # FLOPs -> Flash
    20: "FLOPs_Flash_T4", 21: "FLOPs_Flash_T8", 22: "FLOPs_Flash_T32", 23: "FLOPs_Flash_T16", # Note: T32/T16 swapped in table
}
# 🚨 FIX for swapped columns in the provided data
# The last two columns (22 and 23) in the data blob are T=32 and T=16,
# which is a swap from the previous pattern. We map them correctly:
original_columns_map[22] = "FLOPs_Flash_T32"
original_columns_map[23] = "FLOPs_Flash_T16"


# --- 1. Parse the Data ---
parsed_data = []
for i, line in enumerate(f.readlines()):
    if i >= len(row_labels):
        print(f"Warning: Skipping extra data line {i+1}.")
        continue
    
    parts = line.split()
    
    if len(parts) != 24:
        print(f"Warning: Skipping line {i+1}, expected 24 columns, got {len(parts)}.")
        continue

    row_data = {
        "Model": row_labels[i][0],
        "Flow": row_labels[i][1],
    }
    
    for idx, part_val in enumerate(parts):
        # Clean FLOPs data (remove commas)
        if idx >= 16:
            part_val = part_val.replace(',', '')
        
        # Get the correct column name from our map
        col_name = original_columns_map.get(idx)
        if col_name:
            row_data[col_name] = part_val
        
    parsed_data.append(row_data)

df = pd.DataFrame(parsed_data)

# --- 2. Calculate Energy Costs ---
energy_results = []
for _, row in df.iterrows():
    new_row = {
        "Model": row["Model"],
        "Flow": row["Flow"],
    }
    
    for method in ["Prop", "Flash"]:
        for tile in [4, 8, 16, 32]:
            # Get metric keys
            io_col = f"IO_{method}_T{tile}"
            flops_col = f"FLOPs_{method}_T{tile}"
            
            # Get values (convert to numeric)
            try:
                # 🚨 MODIFICATION: Use the combined IO_ column
                total_io_mb = float(row[io_col]) 
                flops = int(row[flops_col])
            except KeyError as e:
                print(f"Error: Missing column {e} for {row['Model']} {method} T{tile}")
                continue
            except ValueError as e:
                print(f"Error: Invalid data {e} for {row['Model']} {method} T{tile}")
                continue

            # Calculate Energy
            # 🚨 MODIFICATION: Directly use total_io_mb
            dram_energy = (total_io_mb * BITS_PER_MB) * DRAM_COST_PER_BIT
            mac_energy = flops * MAC_COST_PER_FLOP
            
            # Store results (still in nJ)
            new_row[f"DRAM_Energy_{method}_T{tile}"] = dram_energy
            new_row[f"MAC_Energy_{method}_T{tile}"] = mac_energy
    
    energy_results.append(new_row)

df_energy = pd.DataFrame(energy_results)

# --- 3. Format Output Table ---
# Create the desired MultiIndex (Metric -> Method -> Tile)
# We will label the output as (nJ) to match the calculation
metrics = ["DRAM Energy Cost (nJ)", "MAC Energy Cost (nJ)"]
methods = ["Proposed", "Flash"]
tiles = ["T=4", "T=8", "T=16", "T=32"]

new_multi_index_tuples = []
for metric in metrics:
    for method in methods:
        for tile in tiles:
            new_multi_index_tuples.append((metric, method, tile))

# Define the order of the calculated columns to match the new MultiIndex
original_column_order_map = [
    # DRAM Energy -> Proposed
    "DRAM_Energy_Prop_T4", "DRAM_Energy_Prop_T8", "DRAM_Energy_Prop_T16", "DRAM_Energy_Prop_T32",
    # DRAM Energy -> Flash
    "DRAM_Energy_Flash_T4", "DRAM_Energy_Flash_T8", "DRAM_Energy_Flash_T16", "DRAM_Energy_Flash_T32",
    
    # MAC Energy -> Proposed
    "MAC_Energy_Prop_T4", "MAC_Energy_Prop_T8", "MAC_Energy_Prop_T16", "MAC_Energy_Prop_T32",
    # MAC Energy -> Flash
    "MAC_Energy_Flash_T4", "MAC_Energy_Flash_T8", "MAC_Energy_Flash_T16", "MAC_Energy_Flash_T32",
]

# Create the re-ordered data DataFrame
df_data = df_energy[original_column_order_map]

# Create the MultiIndex object
new_multi_index = pd.MultiIndex.from_tuples(new_multi_index_tuples, names=["Metric", "Method", "Tile"])

# Assign the new MultiIndex to the data
df_data.columns = new_multi_index

# Combine with Model/Flow
df_final = pd.concat([df_energy[["Model", "Flow"]], df_data], axis=1)

# Save to CSV
output_filename = "energy_cost_analysis.csv"
# Format as integer (rounding the nJ values)
df_final.to_csv(output_filename, index=False, float_format='%.0f') 

print(f"Successfully calculated energy costs from 24-column data and saved to {output_filename}")
print(df_final.head())
