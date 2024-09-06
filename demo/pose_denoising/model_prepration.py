
import tensorflow_hub as hub
import openvino as ov
import tensorflow as tf
#model = tf.keras.models.load_model("G:\Research\AIBrain\Model\saved_model.pb")

#model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
#ov_model = ov.convert_model(model)

#ov_model = ov.convert_model("G:\Research\AIBrain\Model\saved_model.pb")

import kagglehub

# Download latest version
#path = kagglehub.model_download("google/movenet/tensorFlow2/singlepose-lightning")
#print("Path to model files:", path)
#ov.convert_model(input_model=r"G:\Research\AIBrain\Model\movenet\singlepose-lightning\4\saved_model.pb", input=[192,192,3])

core = ov.Core()
model = core.read_model(r"G:\Models\movenet\singlepose-lightning\4\4.xml")