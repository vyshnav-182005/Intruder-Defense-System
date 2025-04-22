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
import lgpio  # Added for GPIO control

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

# ===================== GPIO Setup =====================
TRIG = 5
ECHO = 6
IN1, IN2 = 17, 27
IN3, IN4 = 22, 23
gpio_handle = lgpio.gpiochip_open(0)

# Setup motor and ultrasonic pins
lgpio.gpio_claim_output(gpio_handle, TRIG, 0)
lgpio.gpio_claim_input(gpio_handle, ECHO)

for pin in [IN1, IN2, IN3, IN4]:
    lgpio.gpio_claim_output(gpio_handle, pin, 0)

# ===================== Motor and Movement =====================
def measure_ultrasonic_distance():
    lgpio.gpio_write(gpio_handle, TRIG, 0)
    time.sleep(0.0002)
    lgpio.gpio_write(gpio_handle, TRIG, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(gpio_handle, TRIG, 0)

    timeout = time.time() + 0.05
    while lgpio.gpio_read(gpio_handle, ECHO) == 0 and time.time() < timeout:
        pass
    start = time.time()
    while lgpio.gpio_read(gpio_handle, ECHO) == 1 and time.time() < timeout:
        pass
    end = time.time()

    duration = end - start
    distance = (duration * 34300) / 2
    return round(distance, 2)

def move_forward():
    print("Moving forward...")
    lgpio.gpio_write(gpio_handle, IN1, 1)
    lgpio.gpio_write(gpio_handle, IN2, 0)
    lgpio.gpio_write(gpio_handle, IN3, 1)
    lgpio.gpio_write(gpio_handle, IN4, 0)
    time.sleep(0.5)
    stop_chassis()

def steer_left():
    print("Obstacle detected: Steering left...")
    lgpio.gpio_write(gpio_handle, IN1, 0)
    lgpio.gpio_write(gpio_handle, IN2, 1)
    lgpio.gpio_write(gpio_handle, IN3, 1)
    lgpio.gpio_write(gpio_handle, IN4, 0)
    time.sleep(0.4)
    stop_chassis()

def steer_right():
    print("Obstacle detected: Steering right...")
    lgpio.gpio_write(gpio_handle, IN1, 1)
    lgpio.gpio_write(gpio_handle, IN2, 0)
    lgpio.gpio_write(gpio_handle, IN3, 0)
    lgpio.gpio_write(gpio_handle, IN4, 1)
    time.sleep(0.4)
    stop_chassis()

def stop_chassis():
    for pin in [IN1, IN2, IN3, IN4]:
        lgpio.gpio_write(gpio_handle, pin, 0)

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
            last_sent_time = current_time
            print(f"Class: {pred_class}, Distance: {distance} cm, Probabilities: {probabilities}")

        # Check for distance from ultrasonic sensor
        ultrasonic_distance = measure_ultrasonic_distance()
        print(f"Measured ultrasonic distance: {ultrasonic_distance} cm")

        if ultrasonic_distance > 40:
            move_forward()
        elif abs(ultrasonic_distance - 6 * distance) > 10:
            if dx > 0:
                steer_left()
            else:
                steer_right()

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
