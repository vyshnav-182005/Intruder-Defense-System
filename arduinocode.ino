#include <Servo.h>
#include <math.h>

Servo baseServo;
Servo tiltServo;
Servo motor;

const int basePin = 9;
const int tiltPin = 8;
const int motorPin = 10;
const int dcMotorPin = 12;

bool motorState = false;
bool dcMotorState = false;
unsigned long lastToggleTime = 0;

void setup() {
  Serial.begin(9600);
  baseServo.attach(basePin);
  tiltServo.attach(tiltPin);
  motor.attach(motorPin);
  pinMode(dcMotorPin, OUTPUT);

  baseServo.write(90);
  tiltServo.write(110);
  motor.write(0);
  digitalWrite(dcMotorPin, HIGH);
}

void loop() {
  motorState = !motorState;
  motor.write(motorState ? 45 : 0);
  delay(1000);

  unsigned long currentTime = millis();
  if (currentTime - lastToggleTime >= 2000) {
    dcMotorState = !dcMotorState;
    digitalWrite(dcMotorPin, dcMotorState ? HIGH : LOW);
    lastToggleTime = currentTime;
  }

  if (Serial.available()) {
    float x = Serial.parseFloat();
    float y = Serial.parseFloat();
    float z = Serial.parseFloat();

    z = -z;
    float magnitude = sqrt(x*x + y*y + z*z);
    if (magnitude > 1.0) {
      x /= magnitude;
      y /= magnitude;
      z /= magnitude;
    }

    float yawAngle = atan2(y, x) * 180.0 / PI;
    yawAngle = constrain(yawAngle + 90, 20, 160);

    float distance = sqrt(x*x + y*y);
    float pitchAngle = atan2(z, distance) * 180.0 / PI;
    pitchAngle = constrain(110 - pitchAngle, 80, 180);

    baseServo.write(yawAngle);
    tiltServo.write(pitchAngle);
  }
}
