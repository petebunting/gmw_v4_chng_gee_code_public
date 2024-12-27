import os
import ee
import geopandas
import datetime
import pb_gee_tools.datasets
import pathlib

ee.Authenticate()
ee.Initialize(project="ee-petebunting-gmw")

def calc_vld_msk(img):
    return img.select('NIR').gt(0).rename('VLD_MSK')

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

year = 2002

start_date = datetime.datetime(year=year, month=1, day=1)
end_date = datetime.datetime(year=year, month=12, day=31)
out_no_data_val = 0.0

out_gdrive_dir = f"gmw_{year}_mng_count_tiles"

gmw_hab_msk_img = ee.Image('projects/ee-petebunting-gmw/assets/gmw_v23_hab_msk')

def apply_gmw_msk(img):
  return img.updateMask(gmw_hab_msk_img)

mdls_created_lut_dir = "gmw_prj_mdls_created"
sub_cls_lut_dir = "gmw_tiles_cls_sub"

prj_rgns_vec_file = "gmw_tiles_prj_def_hab_intersect.geojson"
prj_rgns_vec_lyr = "gmw_tiles_prj_def_hab_intersect"
prjs_col_name = "gmw_prj"
tiles_col_name = "gmw_tile_name"

prj_tiles_gdf = geopandas.read_file(prj_rgns_vec_file, layer=prj_rgns_vec_lyr)
prjs_unq_col = prj_tiles_gdf[prjs_col_name]
prjs_names = prjs_unq_col.unique()

start_prj = 0
n_prjs = len(prjs_names)
print(f"n_prjs: {n_prjs}\n")
end_tile = 60#(n_prjs-start_prj)

prjs_names = prjs_names[start_prj:end_tile]

#prjs_names = ["GMW-09-008", "GMW-04-002", "GMW-05-001", "GMW-01-009"]
#end_tile = len(prjs_names)

n = 1
for prj_name in prjs_names:
    print(f"Processing {prj_name} - {n} of {n_prjs}")
    mdl_n = 1

    mdl_lut_file = os.path.join(mdls_created_lut_dir, f"{prj_name}_mdl_{mdl_n}.txt")
    cls_mdl_asset_id = f'projects/ee-petebunting-gmw/assets/gmw_ls_cls_mdls/{prj_name}_rf_cls_{mdl_n}'
    trained_cls = ee.Classifier.load(cls_mdl_asset_id)
    if os.path.exists(mdl_lut_file):
        prj_sub_gdf = prj_tiles_gdf[prj_tiles_gdf[prjs_col_name] == prj_name]

        tile_names = prj_sub_gdf[tiles_col_name]
        for tile_name in tile_names:
            print(f"\t{tile_name}")
            out_cls_name = f"{tile_name}_{year}_mng_cls_count_{mdl_n}"
            lut_chk_file = os.path.join(sub_cls_lut_dir, f"{out_cls_name}_subd.txt")

            if not os.path.exists(lut_chk_file):
                tile_sub_gdf = prj_sub_gdf[prj_sub_gdf[tiles_col_name] == tile_name]

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

                ls_img_col = ls_img_col.map(apply_gmw_msk)

                ls_indices_img_col = ls_img_col.map(calc_band_indices)
                ls_vld_msk_img_col = ls_img_col.map(calc_vld_msk)

                def apply_cls(img):
                    out_cls = img.classify(trained_cls)
                    mng_msk_img = out_cls.eq(1).mask(out_cls.eq(1)).rename("Mangroves")
                    return mng_msk_img

                cls_mng_imgs = ls_indices_img_col.map(apply_cls)

                cls_mng_img = cls_mng_imgs.reduce(ee.Reducer.sum())#.mask(gmw_hab_msk_img)
                vld_msk_img = ls_vld_msk_img_col.reduce(ee.Reducer.sum())#.mask(gmw_hab_msk_img)
                out_img = ee.ImageCollection([cls_mng_img, vld_msk_img]).toBands().toInt()

                task = ee.batch.Export.image.toDrive(
                    image=out_img,
                    description=out_cls_name,
                    folder=out_gdrive_dir,
                    crs="EPSG:4326",
                    scale=30,
                    region=tile_aoi,
                    fileFormat="GeoTIFF",
                    formatOptions={"cloudOptimized": True, "noData": out_no_data_val},
                )
                task.start()
                pathlib.Path(lut_chk_file).touch()

    print("")
    n += 1
