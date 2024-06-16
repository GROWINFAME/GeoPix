import os

import rasterio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from modules import Pixel2Corrector

app = FastAPI()


@app.post("/api/")
def upload_photo(scene: UploadFile = File(...)):
    """
    Receiving photo and
    Method finds anomalies and saves them in .csv
    """
    # check if scene with right format
    scene_name = scene.filename
    if not scene_name.endswith(".tif"):
        return {"result": "error", "text": "Send picture in '.tif' format"}, 400

    # temporary save scene in docker container
    with open(scene_name, 'wb') as f:
        f.write(scene.file.read())

    # get result
    result_csv = f"{scene_name.rstrip('.tif')}.csv"
    px = Pixel2Corrector(scene_name, result_csv)
    tiff_data = px.correct()
    c, h, w = tiff_data.shape
    res_name = 'restored_' + scene_name
    with rasterio.open(res_name, mode='w', height=h, width=w, driver='GTiff', count=c,
                       dtype=rasterio.uint16) as dst:
        dst.write(tiff_data)

    # delete temporary file
    if os.path.exists(scene_name):
        os.remove(scene_name)

    return FileResponse(res_name)
