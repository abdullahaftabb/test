import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ResumeDataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            self.images = data['images']
            self.categories = {cat['id']: cat['name'] for cat in data['categories']}
            self.annotations = data['annotations']
        
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        anns = self.img_to_anns.get(img_info['id'], [])
        
        boxes = []
        labels = []
        for ann in anns:
            x_min = ann['bbox'][0]
            y_min = ann['bbox'][1]
            x_max = x_min + ann['bbox'][2]
            y_max = y_min + ann['bbox'][3]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info['id']]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(anns),), dtype=torch.int64),
        }

        if self.transform:
            image = self.transform(image)

        return image, target

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(model, data_loader, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader):.4f}')

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = ResumeDataset(
        img_dir='D:\\lki\\layout-model-training-master\\data\\images', 
        annotation_file='D:\\lki\\layout-model-training-master\\data\\result.json', 
        transform=transform
    )
    
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

    num_classes = len(dataset.categories) + 1  # +1 for background class
    model = get_model(num_classes)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Step 1: Train the model
    try:
        print("Starting training...")
        train_model(model, data_loader, optimizer, num_epochs=10, device=device)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save the trained model
    torch.save(model.state_dict(), 'layout_detection_model.pth')
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
