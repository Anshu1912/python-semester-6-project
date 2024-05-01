from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

CLASS_NAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=['*'])

model = tf.keras.models.load_model("../saved_models/saved_model.keras")
# model = tf.keras.layers.TFSMLayer("../saved_models/3", call_endpoint="serving_default")
# model = tf.saved_model.load("../saved_models/3")


@app.get("/ping")
async def ping():
    return "Hello, the server is up and running"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': f"{round(float(confidence)*100, 2)}%"
    }

    

if __name__ == "__main__":
    uvicorn.run(app, port = 8000)
