import os
import ee
import pandas
import geopandas
import pathlib
import glob

ee.Authenticate()
ee.Initialize(project="ee-petebunting-gmw")

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

mdls_created_lut_dir = "gmw_prj_mdls_created"
train_prj_smpls_dir = "gmw_prj_train_smpls"

prj_rgns_vec_file = "gmw_tiles_prj_def.geojson"
prj_rgns_vec_lyr = "gmw_tiles_prj_def"
prjs_col_name = "gmw_prj"

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

    prj_smpls_file = os.path.join(train_prj_smpls_dir, f"{prj_name}_train_smpls.parquet.sz")
    if os.path.exists(prj_smpls_file):
        mdl_lut_files = glob.glob(os.path.join(mdls_created_lut_dir, f"{prj_name}_mdl_*.txt"))
        if len(mdl_lut_files) < 10:
            for i in range(10):
                print(f"\tIteration: {i+1}")
                mdl_lut_file = os.path.join(mdls_created_lut_dir, f"{prj_name}_mdl_{i+1}.txt")
                if not os.path.exists(mdl_lut_file):
                    data_df = pandas.read_parquet(prj_smpls_file)
                    if len(data_df) > 20000:
                        data_df = data_df.sample(n=20000)
                    else:
                        data_df = data_df.sample(frac=0.5)
                    print(f"\t\tThere are {len(data_df)} training samples.")
                    def row_to_feature(row):
                        geometry = None
                        properties = {col: row[col] for col in data_df.columns if col not in ['latitude', 'longitude', '.geo']}
                        return ee.Feature(geometry, properties)

                    # Convert the DataFrame to a list of Features
                    features = data_df.apply(row_to_feature, axis=1).tolist()
                    # Create a FeatureCollection
                    training_data = ee.FeatureCollection(features)
                    print("\t\tTraining Model")
                    trained_cls_mdl = ee.Classifier.smileRandomForest(numberOfTrees=100).train(training_data, "class", bands)

                    print("\t\tSaving trained model")
                    asset_id = f'projects/ee-petebunting-gmw/assets/gmw_ls_cls_mdls/{prj_name}_rf_cls_{i+1}'
                    task = ee.batch.Export.classifier.toAsset(
                        classifier=trained_cls_mdl,
                        description=f'{prj_name}_rf_cls_{i+1}',
                        assetId=asset_id
                    )
                    task.start()
                    pathlib.Path(mdl_lut_file).touch()
                    print("\t\tDone")

    print("")
    n += 1







