from fastapi import FastAPI, File, UploadFile
import os

from modules import PixelCorrector


app = FastAPI()


@app.post("/upload/")
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
    px = PixelCorrector(scene_name, result_csv)
    px.correct()

    # delete temporary file
    if os.path.exists(scene_name):
        os.remove(scene_name)

    return {"result": "ok", "text": f"All results saved in '{result_csv}' file in result folder"}, 200