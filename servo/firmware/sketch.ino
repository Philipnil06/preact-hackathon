// Servo Control — 2x 180° servos on AX22-0031 (P1)
// S1 → P1_IO0, S2 → P1_IO1, S3 → P1_IO2
// Serial commands: "S1:90", "S2:45", "S3:120"

#define S1_PIN P1_IO0
#define S2_PIN P1_IO1
#define S3_PIN P1_IO2

#define SERVO_FREQ   50
#define SERVO_MIN_US 500
#define SERVO_MAX_US 2500
#define PERIOD_US    20000
#define SERVO_BITS   12
#define SERVO_MAX_DUTY 4095

int angle1 = 90;
int angle2 = 90;
int angle3 = 90;

void setServo(int pin, int angle) {
  angle = constrain(angle, 0, 180);
  int pulseUs = map(angle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  int duty = (int)((long)pulseUs * SERVO_MAX_DUTY / PERIOD_US);
  analogWrite(pin, duty);
}

void configureServo(int pin) {
  pinMode(pin, OUTPUT);
  analogWriteFrequency(pin, SERVO_FREQ);
  analogWriteResolution(pin, SERVO_BITS);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  configureServo(S1_PIN);
  configureServo(S2_PIN);
  configureServo(S3_PIN);
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
