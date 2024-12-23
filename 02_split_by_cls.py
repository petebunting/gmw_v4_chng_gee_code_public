
import os
import glob

import rsgislib.tools.filetools


def split_by_attribute(
    vec_file: str,
    vec_lyr: str,
    split_col_name: str,
    multi_layers: bool = True,
    out_vec_file: str = None,
    out_file_path: str = None,
    out_file_ext: str = None,
    out_format: str = "GPKG",
    out_vec_base_pre:str = "",
    out_vec_base_post:str = "",
    dissolve: bool = False,
    chk_lyr_names: bool = True,
):
    """
    A function which splits a vector layer by an attribute value into either
    different layers or different output files.

    :param vec_file: Input vector file
    :param vec_lyr: Input vector layer
    :param split_col_name: The column name by which the vector layer will be split.
    :param multi_layers: Boolean (default True). If True then a mulitple layer output
                         file will be created (e.g., GPKG). If False then individual
                         files will be outputted.
    :param out_vec_file: Output vector file - only used if multi_layers = True
    :param out_file_path: Output file path (directory) if multi_layers = False.
    :param out_file_ext: Output file extension is multi_layers = False
    :param out_format: The output format (e.g., GPKG, ESRI Shapefile).
    :param out_vec_base_pre: a string which is prepended to the output vector file names
    :param out_vec_base_post: a string which is appended to the output vector file names
    :param dissolve: Boolean (Default=False) if True then a dissolve on the specified
                     variable will be run as layers are separated.
    :param chk_lyr_names: If True (default) layer names (from split_col_name) will be
                          checked, which means punctuation removed and all characters
                          being ascii characters.

    """
    import geopandas
    import tqdm

    import rsgislib.tools.utils

    if multi_layers:
        if out_vec_file is None:
            raise rsgislib.RSGISPyException(
                "If a multiple layer output is specified then an output "
                "file needs to be specified to which the layer need to be added."
            )
    if not multi_layers:
        if (out_file_path is None) or (out_file_ext is None):
            raise rsgislib.RSGISPyException(
                "If a single layer output is specified then an output file path "
                "and file extension needs to be specified."
            )

    if "parquet" in os.path.basename(vec_file):
        base_gpdf = geopandas.read_parquet(vec_file)
    else:
        base_gpdf = geopandas.read_file(vec_file, layer=vec_lyr)
    unq_col = base_gpdf[split_col_name]
    unq_vals = unq_col.unique()

    for val in tqdm.tqdm(unq_vals):
        # Subset to value.
        c_gpdf = base_gpdf.loc[base_gpdf[split_col_name] == val]
        # Check for empty or NA geometries.
        c_gpdf = c_gpdf[~c_gpdf.is_empty]
        c_gpdf = c_gpdf[~c_gpdf.isna()]
        # Dissolve if requested.
        if dissolve:
            # Test resolve if an error thrown then it is probably a topological
            # error which can sometimes be solved using a 0 buffer, so try that
            # to see if it works.
            try:
                c_gpdf.dissolve(by=split_col_name)
            except:
                c_gpdf["geometry"] = c_gpdf.buffer(0)
                c_gpdf = c_gpdf.dissolve(by=split_col_name)
        # Write output to disk.
        val_str = f"{val}"
        if multi_layers and (out_format == "GPKG"):
            if chk_lyr_names:
                val_str = rsgislib.tools.utils.check_str(
                    val_str,
                    rm_non_ascii=True,
                    rm_dashs=True,
                    rm_spaces=False,
                    rm_punc=True,
                )
            c_gpdf.to_file(out_vec_file, layer=val_str, driver="GPKG")
        else:
            if chk_lyr_names:
                val_str = rsgislib.tools.utils.check_str(
                    val_str,
                    rm_non_ascii=True,
                    rm_dashs=True,
                    rm_spaces=False,
                    rm_punc=True,
                )

            out_vec_base_pre_tmp = out_vec_base_pre
            if out_vec_base_pre != "":
                out_vec_base_pre_tmp = f"{out_vec_base_pre}_"
            out_vec_base_post_tmp = out_vec_base_post
            if out_vec_base_post != "":
                out_vec_base_post_tmp = f"_{out_vec_base_post}"
            out_vec_lyr = f"{out_vec_base_pre_tmp}{val_str}{out_vec_base_post_tmp}"
            out_vec_file = os.path.join(
                out_file_path, f"{out_vec_lyr}.{out_file_ext}"
            )
            if out_format == "PARQUET":
                out_compress = None
                if "gzip" in out_file_ext:
                    out_compress = "gzip"
                elif "sz" in out_file_ext:
                    out_compress = "snappy"

                c_gpdf.to_parquet(out_vec_file, compression=out_compress)
            elif out_format == "GPKG":
                c_gpdf.to_file(out_vec_file, layer=out_vec_lyr, driver=out_format)
            else:
                c_gpdf.to_file(out_vec_file, driver=out_format)




vec_files = glob.glob("gmw_prj_train_data/*.parquet.sz")

for vec_file in vec_files:
    out_base_name = rsgislib.tools.filetools.get_file_basename(vec_file)
    split_by_attribute(
        vec_file = vec_file,
        vec_lyr = "out_base_name",
        split_col_name = "ref_cls",
        multi_layers = False,
        out_vec_file = None,
        out_file_path = "gmw_prj_train_data_split",
        out_file_ext="parquet.sz",
        out_format="PARQUET",
        out_vec_base_pre = out_base_name,
        out_vec_base_post = "",
        dissolve = False,
        chk_lyr_names = True,
    )

