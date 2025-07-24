from ultralytics import YOLO

if __name__ == '__main__':

    # coco_dataset = 'coco.yaml'
    # aquarium_dataset = 'aquarium.yaml'
    # pet_dataset = 'pet.yaml'

    # yolo11s_coco = YOLO("yolo11s.yaml")
    yolo11n_coco = YOLO("yolo11n.yaml")

    # yolo11s_coco = yolo11s_coco.train(data=".\Datasets\coco.yaml", epochs=25, imgsz=640, workers=2)
    yolo11n_coco = yolo11n_coco.train(data=".\Datasets\coco.yaml", epochs=25, imgsz=640, workers=2)

    # yolo11s_aquarium = YOLO("yolo11s.yaml")
    # yolo11n_aquarium = YOLO("yolo11n.yaml")

    # yolo11s_aquarium = yolo11s_aquarium.train(data=".\\Datasets\\aquarium.v13i.yolov11\\data.yaml", epochs=25, imgsz=640, workers=2)
    # yolo11n_aquarium = yolo11n_aquarium.train(data=".\\Datasets\\aquarium.v13i.yolov11\data.yaml", epochs=25, imgsz=640, workers=2)

    # yolo11s_pet = YOLO("yolo11s.yaml")
    # yolo11n_pet = YOLO("yolo11n.yaml")

    # print(yolo11s_pet.type)
    # yolo11s_pet = yolo11s_pet.train(data=".\Datasets\\Oxford-pet.v6i.yolov11\\data.yaml", epochs=25, imgsz=640, workers=2)
    # yolo11n_pet = yolo11n_pet.train(data=".\Datasets\\Oxford-pet.v6i.yolov11\\data.yaml", epochs=25, imgsz=640, workers=2)
