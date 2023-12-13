from PIL import Image, ImageDraw
import torch
import os
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import pandas as pd
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data['image_path'][idx]
        image = Image.open(image_path).convert("RGB")

        boxes = torch.tensor(self.data['boxes'][idx], dtype=torch.float32)
        labels = torch.tensor(self.data['labels'][idx], dtype=torch.int64)


        if self.transform is not None:
            image = self.transform(image)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        return image, target

def extract_annotations_labels_from_csv(csv_file, image_dir):
    annotations = pd.read_csv(csv_file)
    boxes_list = []
    labels_list = []
    image_paths_list = []
    default_label = 0

    for index, row in annotations.iterrows():
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)

        if 'class_id' in row:
            label = row['class_id']
        else:
            label = default_label

        if label == 0:
            x_min = 0
            y_min = 0
            x_max = 10  
            y_max = 10

            print(f"Image '{image_name}' has class_id 0. Using default bounding box coordinates.")
        elif label == 1:
            region_attributes = json.loads(row['region_shape_attributes'])

            if all(key in region_attributes for key in ['x', 'y', 'width', 'height']) \
                    and all(isinstance(region_attributes[key], (int, float)) and region_attributes[key] >= 0
                            for key in ['x', 'y', 'width', 'height']):
                x_min = region_attributes['x']
                y_min = region_attributes['y']
                x_max = x_min + region_attributes['width']
                y_max = y_min + region_attributes['height']
            else:
                # If the coordinates are missing or invalid, skip appending this image
                print(f"Image '{image_name}' with class_id 1 has invalid bounding box coordinates.")
                continue
        else:
           
            print(f"Image '{image_name}' has a different class_id. Handling can be implemented here.")
            continue

        image_paths_list.append(image_path)
        boxes_list.append([x_min, y_min, x_max, y_max])
        labels_list.append(label)

   
    extracted_data = {'image_path': image_paths_list, 'boxes': boxes_list, 'labels': labels_list}

    return extracted_data

def get_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone, num_classes)
    return model

def train_model(train_data, num_classes, num_epochs=10, batch_size=2, learning_rate=0.001):
    transform = transforms.Compose([transforms.ToTensor()])
    custom_dataset = CustomDataset(train_data, transform=transform)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes).to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):
        for images, targets in data_loader:
            assert isinstance(targets, dict), "Targets should be a dictionary"
            assert 'boxes' in targets.keys() and 'labels' in targets.keys(), "Targets should have keys 'boxes' and 'labels'"
            assert isinstance(targets['boxes'], torch.Tensor) and isinstance(targets['labels'], torch.Tensor), "Boxes and labels should be tensors"
            optimizer.zero_grad()
            images = list(image.to(device) for image in images)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images, targets)
            loss_classifier = criterion(outputs['labels'], torch.cat([t['labels'] for t in targets]))
            loss_box_reg = criterion(outputs['boxes'], torch.cat([t['boxes'] for t in targets]))
            loss = loss_classifier + loss_box_reg

            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item()}")

def predict_and_draw(model, validation_data):
    model.eval()
    for idx, image in enumerate(validation_data):
        with torch.no_grad():
            predictions = model([image])

        img = to_pil_image(image)
        draw = ImageDraw.Draw(img)

        pred_boxes = predictions[0]['boxes']
        pred_labels = predictions[0]['labels']

        for box, label in zip(pred_boxes, pred_labels):
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
            draw.text((box[0], box[1]), f"Label: {label.item()}", fill="red")

        plt.imshow(img)
        plt.axis('off')
        plt.show()

train_csv_file = "C:/Users/rageg/Desktop/Tree identification/Tree images.csv"
train_image_dir = "C:/Users/rageg/Desktop/Hi/Tree images/data/test/tree"
train_annotations_data = extract_annotations_labels_from_csv(train_csv_file, train_image_dir)
num_classes = 2

# Train the model
train_model(train_annotations_data, num_classes)

# Validation
validation_image_dir = "C:/Users/rageg/Desktop/Hi/Tree images/data/val/tree"
validation_image_paths = [os.path.join(validation_image_dir, img) for img in os.listdir(validation_image_dir)]

validation_dataset = CustomDataset({'image_path': validation_image_paths}, transform=transforms.ToTensor())

# Use the trained model for inference on validation data
model = get_model(num_classes)
model.eval()

predict_and_draw(model, validation_dataset)