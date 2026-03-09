# In[]  global station dealing 2.0
# filter 1982-2020

# one month at least 10 days，at least 10 months one year

import pandas as pd
import glob
import os


base_folder = "E:/gsod2/"
base_folder1 = "F:/global_wind/1982-2020merge2/"

selected_columns = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "WDSP", "WDSP_ATTRIBUTES"]


all_years_data = []


for year_folder in sorted(os.listdir(base_folder)):  
    folder_path = os.path.join(base_folder, year_folder)
    

    if not os.path.isdir(folder_path):
        continue

    print(f"Processing {year_folder}...")


    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))


    df_list = [pd.read_csv(file, usecols=selected_columns, parse_dates=["DATE"]) for file in csv_files]
    

    df_all = pd.concat(df_list, ignore_index=True)


    df_all = df_all[df_all["WDSP_ATTRIBUTES"] >= 4]

    # 提取年份和月份
    df_all["YEAR"] = df_all["DATE"].dt.year
    df_all["MONTH"] = df_all["DATE"].dt.month


    valid_months = df_all.groupby(["STATION", "YEAR", "MONTH"])["DATE"].count().reset_index()


    valid_months = valid_months[valid_months["DATE"] >= 10]

  
    valid_years = valid_months.groupby(["STATION", "YEAR"])["MONTH"].count().reset_index()

   
    valid_years = valid_years[valid_years["MONTH"] >= 10]


    df_filtered = df_all.merge(valid_years[["STATION", "YEAR"]], on=["STATION", "YEAR"])

    
    output_path = os.path.join(base_folder1, f"filtered_{year_folder}.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved cleaned data for {year_folder} -> {output_path}")

 
    all_years_data.append(df_filtered)

if all_years_data:
    
    common_stations = set(all_years_data[0]["STATION"])
    for df in all_years_data[1:]:
        common_stations.intersection_update(set(df["STATION"]))  

    print(f"共有 {len(common_stations)} 个 STATION 在所有年份中存在")

    # only keep same station
    all_years_filtered = [df[df["STATION"].isin(common_stations)] for df in all_years_data]

    # merge all data
    df_final = pd.concat(all_years_filtered, ignore_index=True)


    final_output_path = os.path.join(base_folder1, "merged_filtered_common_stations.csv")
    df_final.to_csv(final_output_path, index=False)
    
    print(f"All years merged (only common STATIONs) and saved to {final_output_path}")
    
    
    