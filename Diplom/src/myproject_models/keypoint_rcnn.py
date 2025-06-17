import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import torchvision.transforms as T

_transform = T.Compose([T.ToTensor()])

def load_model(device=None):
    """Загружает предобученную модель Keypoint RCNN"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = keypointrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()
    model.device = device
    return model

def get_keypoints(image, model, device, threshold=0.9):
    """Извлекает ключевые точки и уверенности из изображения"""
    image_tensor = _transform(image).to(device)
    with torch.no_grad():
        output = model([image_tensor])[0]
    if len(output['keypoints']) > 0 and output['scores'][0] > threshold:
        kpts = output['keypoints'][0][:, :2].cpu().numpy()
        conf = output['keypoints'][0][:, 2].cpu().numpy()
        return kpts, conf
    return None, None