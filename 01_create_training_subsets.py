import os
import tqdm


def clip_vec_into_sets(
    vec_file,
    vec_lyr,
    vec_roi_file,
    vec_roi_lyr,
    roi_att_col,
    out_vec_dir,
    out_vec_base_pre,
    out_vec_base_post,
    out_vec_ext,
    out_format="GPKG",
    out_name_lower=False,
):
    import geopandas
    import rsgislib.tools.utils

    if not os.path.exists(out_vec_dir):
        os.mkdir(out_vec_dir)

    print("Reading Data")
    if "parquet" in os.path.basename(vec_file):
        data_gdf = geopandas.read_parquet(vec_file)
    else:
        data_gdf = geopandas.read_file(vec_file, layer=vec_lyr)

    print("Reading ROI Data")
    if "parquet" in os.path.basename(vec_roi_file):
        roi_gdf = geopandas.read_parquet(vec_roi_file)
    else:
        roi_gdf = geopandas.read_file(vec_roi_file, layer=vec_roi_lyr)

    unq_roi_vals = roi_gdf[roi_att_col].unique()

    print(f"There are {len(unq_roi_vals)} unique roi vals")
    print("Running Subsetting:")
    for roi_val in tqdm.tqdm(unq_roi_vals):
        roi_sub_gdf = roi_gdf[roi_gdf[roi_att_col] == roi_val]
        roi_data_gdf = data_gdf.clip(roi_sub_gdf, keep_geom_type=True)

        out_vec_base_pre_tmp = out_vec_base_pre
        if out_vec_base_pre != "":
            out_vec_base_pre_tmp = f"{out_vec_base_pre}_"
        out_vec_base_post_tmp = out_vec_base_post
        if out_vec_base_post != "":
            out_vec_base_post_tmp = f"_{out_vec_base_post}"
        out_vec_lyr = f"{out_vec_base_pre_tmp}{roi_val}{out_vec_base_post_tmp}"
        out_vec_lyr = rsgislib.tools.utils.check_str(
            out_vec_lyr,
            rm_non_ascii=True,
            rm_dashs=False,
            rm_spaces=True,
            rm_punc=True,
        )
        if out_name_lower:
            out_vec_lyr = out_vec_lyr.lower()

        out_vec_file = os.path.join(out_vec_dir, f"{out_vec_lyr}.{out_vec_ext}")

        if out_format == "PARQUET":
            out_compress = None
            if "gzip" in out_vec_ext:
                out_compress = "gzip"
            elif "sz" in out_vec_ext:
                out_compress = "snappy"

            roi_data_gdf.to_parquet(out_vec_file, compression=out_compress)
        elif out_format == "GPKG":
            roi_data_gdf.to_file(out_vec_file, layer=out_vec_lyr, driver=out_format)
        else:
            roi_data_gdf.to_file(out_vec_file, driver=out_format)


prj_rgns_vec_file = "gmw_tiles_prj_def.geojson"
prj_rgns_vec_lyr = "gmw_tiles_prj_def"
prjs_col_name = "gmw_prj"

glb_train_vec_file = "/Users/pete/Dropbox/University/Research/Projects/GlobalMangroveWatch/GMW_v4_Development/gmw_v4_ref_samples_qad_20240109/gmw_v4_ref_smpls_qad_v6.parquet.gzip"
glb_train_vec_lyr = "gmw_v4_ref_smpls_qad_v6"

clip_vec_into_sets(
    glb_train_vec_file,
    glb_train_vec_lyr,
    prj_rgns_vec_file,
    prj_rgns_vec_lyr,
    prjs_col_name,
    out_vec_dir="gmw_prj_train_data",
    out_vec_base_pre="",
    out_vec_base_post="refs_smps",
    out_vec_ext="parquet.sz",
    out_format="PARQUET",
    out_name_lower=True,
)
