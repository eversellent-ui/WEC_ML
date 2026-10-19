import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Template_Content.starter_kit.model_A import Material_Identifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# this is a struct which holds the different ways in which PIL images are transformed
val_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#val_dataset now stores all the pil images extracted from the folder
val_dataset=datasets.ImageFolder(root="Template_Content/ESCORT_DATA_SET/val",transform=val_transform)

output_image:str="confusion matrix.png"
# it feeds the image to model in batches i.e. a collection of 32 images at once as a tensor (32,3,96,96)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)        # its a tuple (image, identity)

class_Names=val_dataset.classes # A list which contains different folders name holding specific class image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=Material_Identifier(num_classes=len(class_Names)).to(device)
model.load_state_dict(torch.load("best_model.pth",map_location=device, weights_only=True))

model.eval()


Test_Img_Container=[]

Predicted_Img_Container=[]

with torch.no_grad():  # with is used for file handling in order to ensure that the file which is opened gets closed even if an error occurs midway in the code here the teardown logic
    for images, labels in val_loader:
        images = images.to(device) # for memory transfer b/w GPU and CPU
        outputs = model(images)    # The output from the CNN comes in unnormalized tensor format called logits
        predics = torch.argmax(outputs, dim=1) 

        Predicted_Img_Container.extend(predics.cpu().tolist())
        Test_Img_Container.extend(labels.tolist())

Matrices_Data=confusion_matrix(Test_Img_Container,Predicted_Img_Container)  # the diagonal entries will represent the no. of images correctly classified by the model and rest any value will represent wrong identification thus providing that the model is getting confused
                                                                            # counts stuff
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=Matrices_Data, display_labels=class_Names)
disp.plot(ax=ax, cmap=plt.cm.Reds, xticks_rotation=45, colorbar=True)

plt.title("Validation Confusion Matrix")
plt.tight_layout()
plt.savefig(output_image, dpi=300)
plt.close()

print(f"Confusion matrix saved to {output_image}")



