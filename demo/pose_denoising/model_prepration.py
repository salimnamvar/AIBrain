
import tensorflow_hub as hub
import openvino as ov
import tensorflow as tf
#model = tf.keras.models.load_model("G:\Research\AIBrain\Model\saved_model.pb")

model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
ov_model = ov.convert_model(model)

#ov_model = ov.convert_model("G:\Research\AIBrain\Model\saved_model.pb")