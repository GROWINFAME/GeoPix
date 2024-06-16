import os

from fastapi import FastAPI, File, UploadFile, Form

from constants import LAYOUT_FOLDER, RESULT_FOLDER
from modules import MainModule

main_module = MainModule(output_save_dir=RESULT_FOLDER)

app = FastAPI()


@app.post("/api/")
def upload_photo(layout: str = Form(...), scene: UploadFile = File(...)):
    """
    Receiving photo and layout name
    Method saves result in .csv file in volume
    and also returns result as a json
    """
    # check if layout exists
    LAYOUT_PATH = os.path.join(LAYOUT_FOLDER, layout)
    if not os.path.exists(LAYOUT_PATH):
        return {"result": "error", "text": f"No layout with name '{layout}'"}

    # check if scene with right format
    scene_name = scene.filename
    if not scene_name.endswith(".tif"):
        return {"result": "error", "text": "Send picture in '.tif' format"}

    # temporary save scene in docker container
    with open(scene_name, 'wb') as f:
        f.write(scene.file.read())

    # get result
    result = main_module.make_matching(scene_path=scene_name,
                                       layout_path=LAYOUT_PATH)

    # delete temporary file
    if os.path.exists(scene_name):
        os.remove(scene_name)

    return result
