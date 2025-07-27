from fastai.learner import load_learner
from fastai.vision.core import PILImage

# Load the exported learner

learn = load_learner("floor_detector2.pkl")

# Prepare and predict
img = PILImage.create(r"DataSet\4\building08_front_day.jpg")
img = img.resize((224, 224)) 
pred_class, pred_idx, probs = learn.predict(img)

print(f"Predicted Class: {pred_class}")
print(f"Class Index: {pred_idx}")
print(f"Probabilities: {probs}")
