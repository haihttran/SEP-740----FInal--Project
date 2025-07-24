from ultralytics import YOLO
from os import listdir
from os.path import isfile, join
import pandas as pd

path = '.\\images\\coco'
# path = '.\\images\\pet'
# path = '.\\images\\aquarium'
imgs = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

# print(imgs)
weights_path = ['.\\weights\\yolo11s-coco.pt','.\\weights\\yolo11n-coco.pt','.\\weights\\yolo11n-coco-distilled.pt']
# weights_path = ['.\\weights\\yolo11s-pet.pt','.\\weights\\yolo11n-pet.pt','.\\weights\\yolo11n-pet-distilled.pt']
# weights_path = ['.\\weights\\yolo11s-aquarium.pt','.\\weights\\yolo11n-aquarium.pt','.\\weights\\yolo11n-aquarium-distilled.pt']
# print('weights ', weights)
outcomes = []
for w in weights_path:
    print(w)
    # Load a model
    model = YOLO(w)  # pretrained YOLO11n model

    # Run batched inference on a list of images
    results = model(imgs, save=True)  # return a list of Results objects
    
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        # result.save(filename="result.jpg")  # save to disk
        df = result.to_df()
        df['image_name'] = result.path
        df['model'] = str(w)
        outcomes.append(df)
        # print(result.to_df())

df = pd.concat(outcomes)
# print(df)
# df.to_csv('./aquarium_inference_result.csv')
agg_df = df.groupby(['name','image_name','model'])['class'].count()
# agg_df.to_csv('./aquarium_inf_agg.csv')
print(agg_df)