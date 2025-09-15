import argparse
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ---- โหลดโมเดล HopeNet ----
class Hopenet(torch.nn.Module):
    def __init__(self):
        super(Hopenet, self).__init__()
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = torch.nn.Linear(2048, 66 * 3)  # 66 bins per angle, 3 angles

    def forward(self, x):
        return self.backbone(x)

def get_angle_from_output(output):
    # 66 bins, center at 0, bin width 3 degrees
    idx_tensor = torch.arange(66).float()
    idx_tensor = idx_tensor.unsqueeze(0).to(output.device)

    yaw_predicted = torch.sum(torch.softmax(output[:, 0:66], dim=1) * idx_tensor, dim=1) * 3 - 99
    pitch_predicted = torch.sum(torch.softmax(output[:, 66:132], dim=1) * idx_tensor, dim=1) * 3 - 99
    roll_predicted = torch.sum(torch.softmax(output[:, 132:198], dim=1) * idx_tensor, dim=1) * 3 - 99

    return yaw_predicted, pitch_predicted, roll_predicted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()

    # โหลดโมเดล
    model = Hopenet()
    # checkpoint = torch.load(args.snapshot, map_location=torch.device('cpu'))
    checkpoint = torch.load(args.snapshot, map_location=torch.device('cpu'), weights_only=False)

    model.load_state_dict(checkpoint)
    model.eval()

    # เตรียมภาพ
    img = Image.open(args.image).convert('RGB')
    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transformations(img).unsqueeze(0)

    # ประมวลผล
    with torch.no_grad():
        output = model(img_tensor)
        yaw, pitch, roll = get_angle_from_output(output)

    print(f"Yaw: {yaw.item():.2f}, Pitch: {pitch.item():.2f}, Roll: {roll.item():.2f}")

    # วาดภาพผลลัพธ์
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    center_x, center_y = img_cv.shape[1] // 2, img_cv.shape[0] // 2

    yaw_rad = np.radians(yaw.item())
    pitch_rad = np.radians(pitch.item())

    length = 100
    end_x = int(center_x + length * np.sin(yaw_rad))
    end_y = int(center_y - length * np.sin(pitch_rad))

    cv2.line(img_cv, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2)
    cv2.imwrite('output.jpg', img_cv)
    print("Saved output.jpg with head pose visualization.")

if __name__ == '__main__':
    main()
