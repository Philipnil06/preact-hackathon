// Servo Control — 2x 180° servos on AX22-0031
// Official AX22-0031 mapping: S1 → GPIO 1, S2 → GPIO 14, S3 → GPIO 41
// Serial commands: "S1:90", "S2:45", or "S3:120"

#define S1_PIN 1
#define S2_PIN 14
#define S3_PIN 41

#define SERVO_FREQ   50
#define SERVO_MIN_US 500
#define SERVO_MAX_US 2400
#define PERIOD_US    20000

int angle1 = 90;
int angle2 = 90;
int angle3 = 90;

void setServo(int pin, int angle) {
  angle = constrain(angle, 0, 180);
  int pulseUs = map(angle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  analogWriteFrequency(pin, SERVO_FREQ);
  analogWriteResolution(pin, 16);
  int duty = (int)((long)pulseUs * 65535 / PERIOD_US);
  analogWrite(pin, duty);
}

void setup() {
  Serial.begin(115200);
  pinMode(S1_PIN, OUTPUT);
  pinMode(S2_PIN, OUTPUT);
  pinMode(S3_PIN, OUTPUT);
  setServo(S1_PIN, angle1);
  setServo(S2_PIN, angle2);
  setServo(S3_PIN, angle3);
  Serial.println("READY");
  Serial.print("S1:"); Serial.println(angle1);
  Serial.print("S2:"); Serial.println(angle2);
  Serial.print("S3:"); Serial.println(angle3);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.startsWith("S1:")) {
      angle1 = line.substring(3).toInt();
      setServo(S1_PIN, angle1);
      Serial.print("S1:"); Serial.println(angle1);
    } else if (line.startsWith("S2:")) {
      angle2 = line.substring(3).toInt();
      setServo(S2_PIN, angle2);
      Serial.print("S2:"); Serial.println(angle2);
    } else if (line.startsWith("S3:")) {
      angle3 = line.substring(3).toInt();
      setServo(S3_PIN, angle3);
      Serial.print("S3:"); Serial.println(angle3);
    }
  }
}
