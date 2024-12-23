import os
import pandas
import geopandas
import tqdm

import rsgislib.tools.filetools

train_data_dir = "gmw_prj_train_data_split"
train_csv_smpls_dir = "gmw_tile_smpls_csv_files"
out_smpls_dir = "gmw_prj_train_smpls"

prj_rgns_vec_file = "gmw_tiles_prj_def.geojson"
prj_rgns_vec_lyr = "gmw_tiles_prj_def"
prjs_col_name = "gmw_prj"
tiles_col_name = "gmw_tile_name"

prj_tiles_gdf = geopandas.read_file(prj_rgns_vec_file, layer=prj_rgns_vec_lyr)
prjs_unq_col = prj_tiles_gdf[prjs_col_name]
prjs_names = prjs_unq_col.unique()

start_prj = 0
n_prjs = len(prjs_names)
print(f"n_prjs: {n_prjs}\n")
end_tile = (n_prjs-start_prj)

prjs_names = prjs_names[start_prj:end_tile]
#prjs_names = ["GMW-01-006"]
n = 1
for prj_name in tqdm.tqdm(prjs_names):
    #print(f"Processing {prj_name} - {n} of {n_prjs}")

    out_prj_smpls_file = os.path.join(out_smpls_dir, f"{prj_name}_train_smpls.parquet.sz")
    if not os.path.exists(out_prj_smpls_file):
        prj_sub_gdf = prj_tiles_gdf[prj_tiles_gdf[prjs_col_name] == prj_name]

        vec_mng_smpls_file = os.path.join(
            train_data_dir, f"{prj_name}_refs_smps_1.parquet.sz"
        )
        vec_wtr_smpls_file = os.path.join(
            train_data_dir, f"{prj_name}_refs_smps_2.parquet.sz"
        )
        vec_oth_smpls_file = os.path.join(
            train_data_dir, f"{prj_name}_refs_smps_3.parquet.sz"
        )

        if (
                os.path.exists(vec_mng_smpls_file)
                and os.path.exists(vec_wtr_smpls_file)
                and os.path.exists(vec_oth_smpls_file)
        ):
            tile_smpls_lst = list()
            tile_names = prj_sub_gdf[tiles_col_name]
            for tile_name in tile_names:
                tile_smpls_file_name = f"{tile_name}_cls_smpls"
                tile_smpls_file = os.path.join(train_csv_smpls_dir, f"{tile_smpls_file_name}.csv")
                #print(f"Processing {tile_smpls_file}")
                if os.path.exists(tile_smpls_file):
                    file_size = rsgislib.tools.filetools.get_file_size(tile_smpls_file)
                    #print(file_size)
                    if file_size > 100:
                        tile_smpls_lst.append(pandas.read_csv(tile_smpls_file))

            if len(tile_smpls_lst) > 0:
                gmw_prj_smpls_df = pandas.concat(tile_smpls_lst)
                #print(f"\t{len(gmw_prj_smpls_df)} samples")
                if len(gmw_prj_smpls_df) > 100000:
                    gmw_prj_smpls_df = gmw_prj_smpls_df.sample(n=100000, random_state=42)

                gmw_prj_smpls_df.to_parquet(out_prj_smpls_file, compression="snappy")

    #print("")
    n += 1
