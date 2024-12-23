import os
import ee
import geopandas
import datetime
import pb_gee_tools.datasets
import pb_gee_tools.convert_types

ee.Authenticate()
ee.Initialize(project="ee-petebunting-gmw")


def _make_float(img):
    return img.float()


def calc_band_indices(img):
    img = img.multiply(0.0001).float()
    ndvi = img.normalizedDifference(["NIR", "Red"]).rename("NDVI")
    ndwi = img.normalizedDifference(["NIR", "SWIR1"]).rename("NDWI")
    nbr = img.normalizedDifference(["NIR", "SWIR2"]).rename("NBR")
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR": img.select("NIR"),
            "RED": img.select("Red"),
            "BLUE": img.select("Blue"),
        },
    ).rename(["EVI"])
    mvi = img.expression(
        "((NIR - GREEN) / (SWIR - GREEN))",
        {
            "NIR": img.select("NIR"),
            "SWIR": img.select("SWIR1"),
            "GREEN": img.select("Green"),
        },
    ).rename(["MVI"])

    return img.addBands([ndvi, ndwi, nbr, evi, mvi])


bands = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
    "NDVI",
    "NDWI",
    "NBR",
    "EVI",
    "MVI",
]

start_date = datetime.datetime(year=2020, month=1, day=1)
end_date = datetime.datetime(year=2020, month=12, day=31)
out_no_data_val = 0.0

# gmw_hab_msk_img = ee.Image('projects/ee-petebunting-gmw/assets/gmw_v23_hab_msk')

train_data_dir = "gmw_prj_train_data_split"
train_csv_smpls_dir = "gmw_tile_smpls_csv_files"

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

n = 1
for prj_name in prjs_names:
    print(f"Processing {prj_name} - {n} of {n_prjs}")

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
        mng_pts_gdf = geopandas.read_parquet(vec_mng_smpls_file)
        wat_pts_gdf = geopandas.read_parquet(vec_wtr_smpls_file)
        oth_pts_gdf = geopandas.read_parquet(vec_oth_smpls_file)

        n_mng_pts = len(mng_pts_gdf)
        n_wat_pts = len(wat_pts_gdf)
        n_oth_pts = len(oth_pts_gdf)

        min_n_pts = min(n_mng_pts, n_wat_pts)
        min_n_pts = min(min_n_pts, n_oth_pts)
        if min_n_pts > 10000:
            min_n_pts = 10000

        print(
            f"min_n_pts: {min_n_pts} (mng: {n_mng_pts}, wtr: {n_wat_pts}, oth: {n_oth_pts})"
        )

        mng_pts_gdf = mng_pts_gdf.sample(min_n_pts, random_state=42)
        wat_pts_gdf = wat_pts_gdf.sample(min_n_pts, random_state=42)
        oth_pts_gdf = oth_pts_gdf.sample(min_n_pts, random_state=42)

        tile_names = prj_sub_gdf[tiles_col_name]
        for tile_name in tile_names:
            print(f"\t{tile_name}")
            out_file_name = f"{tile_name}_cls_smpls"
            lcl_csv_file = os.path.join(train_csv_smpls_dir, f"{out_file_name}.csv")
            if not os.path.exists(lcl_csv_file):
                tile_sub_gdf = prj_sub_gdf[prj_sub_gdf[tiles_col_name] == tile_name]

                mng_pts_sub_gdf = mng_pts_gdf.clip(tile_sub_gdf)
                wat_pts_sub_gdf = wat_pts_gdf.clip(tile_sub_gdf)
                oth_pts_sub_gdf = oth_pts_gdf.clip(tile_sub_gdf)

                n_mng_pts_sub = len(mng_pts_sub_gdf)
                n_wat_pts_sub = len(wat_pts_sub_gdf)
                n_oth_pts_sub = len(oth_pts_sub_gdf)

                if (n_mng_pts_sub > 0) or (n_wat_pts_sub > 0) or (n_oth_pts_sub > 0):

                    if n_mng_pts_sub > 0:
                        gee_mng_pts = pb_gee_tools.convert_types.get_gee_pts_gp_gdf(
                            mng_pts_sub_gdf
                        )

                    if n_wat_pts_sub > 0:
                        gee_wtr_pts = pb_gee_tools.convert_types.get_gee_pts_gp_gdf(
                            wat_pts_sub_gdf
                        )

                    if n_oth_pts_sub > 0:
                        gee_oth_pts = pb_gee_tools.convert_types.get_gee_pts_gp_gdf(
                            oth_pts_sub_gdf
                        )

                    if (n_mng_pts_sub > 0) and (n_wat_pts_sub > 0) and (n_oth_pts_sub > 0):
                        train_smpls = ee.FeatureCollection(
                            [
                                ee.Feature(gee_mng_pts, {"class": 1}),
                                ee.Feature(gee_wtr_pts, {"class": 2}),
                                ee.Feature(gee_oth_pts, {"class": 3}),
                            ]
                        )
                    elif (n_mng_pts_sub > 0) and (n_wat_pts_sub > 0):
                        train_smpls = ee.FeatureCollection(
                            [
                                ee.Feature(gee_mng_pts, {"class": 1}),
                                ee.Feature(gee_wtr_pts, {"class": 2}),
                            ]
                        )
                    elif (n_mng_pts_sub > 0) and (n_oth_pts_sub > 0):
                        train_smpls = ee.FeatureCollection(
                            [
                                ee.Feature(gee_mng_pts, {"class": 1}),
                                ee.Feature(gee_oth_pts, {"class": 3}),
                            ]
                        )
                    elif (n_wat_pts_sub > 0) and (n_oth_pts_sub > 0):
                        train_smpls = ee.FeatureCollection(
                            [
                                ee.Feature(gee_wtr_pts, {"class": 2}),
                                ee.Feature(gee_oth_pts, {"class": 3}),
                            ]
                        )
                    elif n_mng_pts_sub > 0:
                        train_smpls = ee.FeatureCollection(
                            [
                                ee.Feature(gee_mng_pts, {"class": 1}),
                            ]
                        )
                    elif n_wat_pts_sub > 0:
                        train_smpls = ee.FeatureCollection(
                            [
                                ee.Feature(gee_wtr_pts, {"class": 2}),
                            ]
                        )
                    elif n_oth_pts_sub > 0:
                        train_smpls = ee.FeatureCollection(
                            [
                                ee.Feature(gee_oth_pts, {"class": 3}),
                            ]
                        )

                    # Get layer bbox: minx, miny, maxx, maxy
                    tile_bbox = tile_sub_gdf.total_bounds

                    # Create the GEE geometry from the bbox.
                    roi_west = tile_bbox[0]
                    roi_east = tile_bbox[2]
                    roi_north = tile_bbox[3]
                    roi_south = tile_bbox[1]
                    tile_aoi = ee.Geometry.BBox(roi_west, roi_south, roi_east, roi_north)

                    ls_img_col = pb_gee_tools.datasets.get_sr_landsat_collection(
                        aoi=tile_aoi,
                        start_date=start_date,
                        end_date=end_date,
                        cloud_thres=70,
                        ignore_ls7=False,
                        out_lstm_bands=True,
                    ).select(["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])

                    sen2_ls_indices_img_col = ls_img_col.map(calc_band_indices)

                    def sample_img_training(img):
                        training = img.sampleRegions(
                            collection=train_smpls, properties=["class"], scale=30
                        )
                        return training

                    training_data = sen2_ls_indices_img_col.map(sample_img_training)
                    training_data = training_data.flatten()

                    n_smples = int(training_data.size().getInfo())
                    print(f"\t\tThere are {n_smples} training samples.")

                    task = ee.batch.Export.table.toDrive(
                        collection=training_data,
                        folder="gmw_v4_chng_2020_ls_train_smpls",
                        description=out_file_name,
                        fileNamePrefix=out_file_name,
                        fileFormat="CSV",
                    )
                    task.start()
    print("")
    n += 1
