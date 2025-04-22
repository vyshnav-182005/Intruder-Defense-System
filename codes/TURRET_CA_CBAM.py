import cv2
import dlib
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import serial
import time
import math
import csv

# ===================== Model Definition =====================

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

# ===================== Setup =====================

num_classes = 4
model = FaceRecognitionModel(num_classes)
model.load_state_dict(torch.load("/home/raspberrypi/Intruder-Defense-System/CA_CBAM.pth", map_location=torch.device('cpu')))
model.eval()

last_sent_time = 0  # Track last sent time
send_interval = 1  # seconds

ser = serial.Serial("/dev/ttyUSB0", 9600)
time.sleep(2)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

CAMERA_OFFSET_Y = 5.0  # camera is 5 cm above the firing point

# ===================== CSV Setup =====================
csv_file = '/home/raspberrypi/face_data_log.csv'
csv_headers = ['OffsetX (cm)', 'OffsetY (cm)', 'Distance (cm)', 'Fire (0 or 1)', 'Class', 'Probabilities']

# Create CSV file with headers if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)

# ===================== Main Loop =====================

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
        dx = face_cx - (frame_w // 2)
        dy = face_cy - (frame_h // 2)

        K = 50.0
        real_face_height = 22.0
        distance = K * real_face_height / h

        fov_degrees = 60
        fov_radians = math.radians(fov_degrees)
        view_width_cm = 2 * distance * math.tan(fov_radians / 2)

        offset_x_cm = (dx / frame_w) * view_width_cm
        offset_y_cm = (dy / frame_w) * view_width_cm

        # Compensate for camera being above the firing point
        offset_y_cm -= CAMERA_OFFSET_Y

        FIRE = 1 if probabilities[pred_class] < 0.95 else 0

        # Log data to CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([offset_x_cm, offset_y_cm, distance, FIRE, pred_class, probabilities])

        current_time = time.time()
        if current_time - last_sent_time >= send_interval:
            FIRE=0
            ser.write(f"{-5*offset_x_cm},{5*offset_y_cm},{distance},{FIRE}\n".encode())
            last_sent_time = current_time

        # Draw overlays
        cv2.circle(frame, (face_cx, face_cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Offset: ({offset_x_cm:.2f}, {offset_y_cm:.2f}) cm", (x, y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(frame, f"Class: {pred_class}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, prob in enumerate(probabilities):
            text = f"class {i}: {prob:.4f}"
            cv2.putText(frame, text, (x, y + h + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("Face Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()

