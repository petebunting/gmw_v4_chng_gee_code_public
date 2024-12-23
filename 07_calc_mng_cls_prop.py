import glob
import os

import rsgislib
import rsgislib.imagecalc
import rsgislib.imageutils
import rsgislib.tools.filetools

rsgislib.imageutils.set_env_vars_lzw_gtiff_outs()

years = ["2000"]
base_dir = "/Users/pfb/Temp/gmw_v4_gee_cls_rslts/"

for year in years:
    imgs_dir = os.path.join(base_dir, year, "counts")
    out_dir = os.path.join(base_dir, year, "prop")

    imgs = glob.glob(os.path.join(imgs_dir, "*.tif"))
    for img in imgs:
        basename = rsgislib.tools.filetools.get_file_basename(img)
        out_img_file = os.path.join(out_dir, f"{basename}_prop.tif")
        if not os.path.exists(out_img_file):
            band_defs = list()
            band_defs.append(
                rsgislib.imagecalc.BandDefn(band_name="mng", input_img=img, img_band=1)
            )
            band_defs.append(
                rsgislib.imagecalc.BandDefn(band_name="vld", input_img=img, img_band=2)
            )
            exp = "(mng/vld)*100"
            rsgislib.imagecalc.band_math(
                out_img_file,
                exp=exp,
                gdalformat="GTIFF",
                datatype=rsgislib.TYPE_8UINT,
                band_defs=band_defs,
            )

            rsgislib.imageutils.pop_img_stats(
                out_img_file, use_no_data=True, no_data_val=0, calc_pyramids=True
            )
