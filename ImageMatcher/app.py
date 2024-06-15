from fastapi import FastAPI, File, UploadFile, Form
from main_module import MainModule
from constants import LAYOUT_FOLDER, RESULT_FOLDER, SCENE_PATH

main_module = MainModule(output_save_dir=RESULT_FOLDER)

app = FastAPI()


@app.post("/upload/")
def upload_photo(layout: str = Form(...), photo: UploadFile = File(...)):
    """
    Receiving photo and layout name
    Method saves result in .csv file in volume
    and also returns result as a json
    """
    
    if not photo.filename.endswith(".tif"):
        return {"result": "error", "text": "Send photo in '.tif' format"}


    with open(SCENE_PATH, 'wb') as f:
        f.write(photo.file.read())

    LAYOUT_PATH = f"{LAYOUT_FOLDER}/{layout}"
    result = main_module.make_matching(scene_path=SCENE_PATH,
                                       layout_path=LAYOUT_PATH)
    
    # delete

    return {"result": result, "text": "All results also saved in coords.csv"}