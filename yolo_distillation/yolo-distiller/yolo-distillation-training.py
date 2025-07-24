from ultralytics import YOLO
import os

if __name__ == '__main__':

    teacher_model_1 = YOLO('..\\weights\\yolo11s-coco.pt')

    student_model_1 = YOLO("yolo11n.pt")

    student_model_1.train(
        data='.\\Datasets\\coco.yaml',
        teacher=teacher_model_1.model, # None if you don't wanna use knowledge distillation
        distillation_loss="cwd",
        device=0,
        epochs=25,
        workers=2,
        # resume=True,
        name='coco',
        exist_ok=True,
    )

    # teacher_model_2 = YOLO('..\\weights\\yolo11s-pet.pt')

    # student_model_2 = YOLO("yolo11n.pt")

    # student_model_2.train(
    #     data="..\\datasets\\Oxford-pet.v6i.yolov11\\data.yaml",
    #     teacher=teacher_model_2.model, # None if you don't wanna use knowledge distillation
    #     distillation_loss="cwd",
    #     device=0,
    #     epochs=25,
    #     workers=2,
    #     name='pet',
    #     exist_ok=True,
    # )

    # teacher_model_3 = YOLO('..\\weights\\yolo11s-aquarium.pt')

    # student_model_3 = YOLO("yolo11n.pt")

    # student_model_3.train(
    #     data='..\\datasets\\aquarium.v13i.yolov11\\data.yaml',
    #     teacher=teacher_model_3.model, # None if you don't wanna use knowledge distillation
    #     distillation_loss="cwd",
    #     device=0,
    #     epochs=25,
    #     workers=2,
    #     name='aquarium',
    #     exist_ok=True,
    # )