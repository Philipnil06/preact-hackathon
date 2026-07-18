// Servo Control — 2x 180° servos on AX22-0031 (P1)
// S1 → P1_IO0, S2 → P1_IO1
// Serial commands: "S1:90" or "S2:45"

#define S1_PIN P1_IO0
#define S2_PIN P1_IO1

#define SERVO_FREQ   50
#define SERVO_MIN_US 500
#define SERVO_MAX_US 2400
#define PERIOD_US    20000

int angle1 = 90;
int angle2 = 90;

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
  setServo(S1_PIN, angle1);
  setServo(S2_PIN, angle2);
  Serial.println("READY");
  Serial.print("S1:"); Serial.println(angle1);
  Serial.print("S2:"); Serial.println(angle2);
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
    }
  }
}