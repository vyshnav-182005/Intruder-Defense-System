import cv2
import dlib
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import serial
import time

# Define the same model architecture as used during training
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).permute(0, 2, 1)
        x2 = x2.flatten(2).permute(0, 2, 1)

        qkv1 = self.qkv(x1).chunk(3, dim=-1)
        qkv2 = self.qkv(x2).chunk(3, dim=-1)

        q, k, v = qkv1[0], qkv2[1], qkv2[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = self.proj(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.cross_attention = CrossAttention(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ca_out = self.cross_attention(x, x)
        return self.sigmoid(ca_out) * x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.cross_attention = CrossAttention(in_channels)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sa_out = self.cross_attention(x, x)
        sa_out = self.conv(sa_out)
        return self.sigmoid(sa_out) * x

class CA_CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CA_CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            
            CA_CBAM(128),
            
            nn.AdaptiveAvgPool2d(1)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Load Model and Weights
num_classes = 4
model = FaceRecognitionModel(num_classes)
model.load_state_dict(torch.load(r"/home/raspberrypi/Intruder-Defense-System/CA_CBAM.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

detector = dlib.get_frontal_face_detector()

# Initialize Serial
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(face_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().numpy()
            pred_class = torch.argmax(output, dim=1).item()

        face_cx = x + w // 2
        face_cy = y + h // 2
        frame_h, frame_w = frame.shape[:2]
        dx = face_cx - frame_w // 2
        dy = face_cy - frame_h // 2

        K = 273.0
        real_face_height = 22.0
        distance = K * real_face_height / h

        fov_degrees = 60
        fov_radians = fov_degrees * (3.14159265 / 180)
        view_width_cm = 2 * distance * (torch.tan(torch.tensor(fov_radians / 2)))

        offset_x_cm = (dx / frame_w) * view_width_cm.item()
        offset_y_cm = (dy / frame_w) * view_width_cm.item()

        norm_x = max(min(offset_x_cm / 50, 1.0), -1.0)
        norm_y = max(min(offset_y_cm / 50, 1.0), -1.0)
        norm_z = max(min(distance / 100.0, 1.0), -1.0)

        serial_msg = f"{norm_x:.2f} {norm_y:.2f} {norm_z:.2f}\n"
        ser.write(serial_msg.encode())

        cv2.putText(frame, f"Class: {pred_class}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Offset: ({offset_x_cm:.2f}, {offset_y_cm:.2f}) cm", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
cap.release()
cv2.destroyAllWindows()
