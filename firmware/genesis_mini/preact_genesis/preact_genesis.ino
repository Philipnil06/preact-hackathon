/*
 * PREACT Genesis Mini firmware
 *
 * Streams VL53L0X Time-of-Flight readings to the laptop and accepts
 * newline-delimited JSON commands that point one hobby servo.
 *
 * Target board: Axiometa Genesis Mini (ESP32-S3)
 * ToF module:   Axiometa AX22-0015 (VL53L0X)
 * Servo module: Axiometa AX22-0031, first servo header
 */

#include <Adafruit_VL53L0X.h>
#include <Adafruit_NeoPixel.h>

// Fixed Genesis Mini port plan:
//   Port 1: AX22-0015 ToF (I2C)
//   Port 2: AX22-0036 TRACKING indicator, Data-In is IO1 / GPIO 6
//   Port 3: AX22-0036 LANDING indicator, Data-In is IO1 / GPIO 16
//   Port 4: AX22-0031 servo adapter, first header is IO0 / GPIO 1
#ifndef PREACT_SERVO_PIN
#define PREACT_SERVO_PIN P4_IO0
#endif
#define PREACT_TRACKING_LED_PIN P2_IO1
#define PREACT_LANDING_LED_PIN P3_IO1
#define PREACT_LED_BRIGHTNESS 48

#define PREACT_PROTOCOL_VERSION 1
#define PREACT_FIRMWARE_VERSION "0.1.0"

#define SERVO_FREQUENCY_HZ 50
#define SERVO_PWM_BITS 12
#define SERVO_PWM_MAX ((1 << SERVO_PWM_BITS) - 1)

// Tune these for the servo on the day. Start conservatively to avoid hard stops.
#define SERVO_MIN_ANGLE_DEG 10.0f
#define SERVO_MAX_ANGLE_DEG 170.0f
#define SERVO_NEUTRAL_ANGLE_DEG 90.0f
#define SERVO_MIN_PULSE_US 600
#define SERVO_MAX_PULSE_US 2400

#define SERIAL_BAUD_RATE 115200
#define MAX_COMMAND_LENGTH 256
#define SENSOR_RETRY_INTERVAL_MS 2000
#define INVALID_SAMPLE_INTERVAL_MS 250

Adafruit_VL53L0X tof;
Adafruit_NeoPixel trackingIndicator(
  1,
  PREACT_TRACKING_LED_PIN,
  NEO_GRB + NEO_KHZ800
);
Adafruit_NeoPixel landingIndicator(
  1,
  PREACT_LANDING_LED_PIN,
  NEO_GRB + NEO_KHZ800
);
bool tofReady = false;
unsigned long lastSensorAttemptMs = 0;
unsigned long lastInvalidSampleMs = 0;
char commandBuffer[MAX_COMMAND_LENGTH];
size_t commandLength = 0;

static float clampFloat(float value, float lower, float upper) {
  if (value < lower) return lower;
  if (value > upper) return upper;
  return value;
}

static int angleToPulseUs(float angleDeg) {
  const float bounded = clampFloat(
    angleDeg,
    SERVO_MIN_ANGLE_DEG,
    SERVO_MAX_ANGLE_DEG
  );
  const float normalized = bounded / 180.0f;
  return (int)(
    SERVO_MIN_PULSE_US
    + normalized * (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US)
  );
}

static int pulseUsToDuty(int pulseUs) {
  const int periodUs = 1000000 / SERVO_FREQUENCY_HZ;
  const long duty = (long)pulseUs * SERVO_PWM_MAX / periodUs;
  if (duty < 0) return 0;
  if (duty > SERVO_PWM_MAX) return SERVO_PWM_MAX;
  return (int)duty;
}

static void writeServoAngle(float angleDeg) {
  analogWrite(
    PREACT_SERVO_PIN,
    pulseUsToDuty(angleToPulseUs(angleDeg))
  );
}

static const char *findFieldValue(const char *json, const char *field) {
  static char pattern[40];
  snprintf(pattern, sizeof(pattern), "\"%s\"", field);
  const char *position = strstr(json, pattern);
  if (position == nullptr) return nullptr;
  position += strlen(pattern);
  while (*position == ' ' || *position == '\t') position++;
  if (*position != ':') return nullptr;
  position++;
  while (*position == ' ' || *position == '\t') position++;
  return position;
}

static bool readLongField(const char *json, const char *field, long *value) {
  const char *position = findFieldValue(json, field);
  if (position == nullptr) return false;
  char *end = nullptr;
  const long parsed = strtol(position, &end, 10);
  if (end == position) return false;
  *value = parsed;
  return true;
}

static bool readFloatField(const char *json, const char *field, float *value) {
  const char *position = findFieldValue(json, field);
  if (position == nullptr) return false;
  char *end = nullptr;
  const float parsed = strtof(position, &end);
  if (end == position || !isfinite(parsed)) return false;
  *value = parsed;
  return true;
}

static bool messageHasType(const char *json, const char *type) {
  char pattern[64];
  snprintf(pattern, sizeof(pattern), "\"type\":\"%s\"", type);
  if (strstr(json, pattern) != nullptr) return true;

  // Also accept a single space after the colon for hand-written commands.
  snprintf(pattern, sizeof(pattern), "\"type\": \"%s\"", type);
  return strstr(json, pattern) != nullptr;
}

static bool messageHasStringField(
  const char *json,
  const char *field,
  const char *value
) {
  char pattern[80];
  snprintf(pattern, sizeof(pattern), "\"%s\":\"%s\"", field, value);
  if (strstr(json, pattern) != nullptr) return true;
  snprintf(pattern, sizeof(pattern), "\"%s\": \"%s\"", field, value);
  return strstr(json, pattern) != nullptr;
}

static void setModeIndicators(const char *mode) {
  trackingIndicator.clear();
  landingIndicator.clear();
  if (strcmp(mode, "tracking") == 0) {
    trackingIndicator.setPixelColor(
      0,
      trackingIndicator.Color(0, 255, 0)
    );
  } else if (strcmp(mode, "landing") == 0) {
    landingIndicator.setPixelColor(
      0,
      landingIndicator.Color(255, 0, 0)
    );
  }
  trackingIndicator.show();
  landingIndicator.show();
}

static void sendAck(long commandId, bool accepted, const char *detail) {
  Serial.printf(
    "{\"type\":\"ack\",\"command_id\":%ld,\"accepted\":%s,"
    "\"detail\":\"%s\"}\n",
    commandId,
    accepted ? "true" : "false",
    detail
  );
}

static bool protocolVersionIsValid(const char *json) {
  long version = 0;
  return readLongField(json, "protocol_version", &version)
    && version == PREACT_PROTOCOL_VERSION;
}

static void processCommand(const char *json) {
  long commandId = -1;
  readLongField(json, "command_id", &commandId);

  if (messageHasType(json, "ping")) {
    long nonce = 0;
    readLongField(json, "nonce", &nonce);
    sendAck(nonce, true, "pong");
    return;
  }

  if (messageHasType(json, "mode")) {
    if (!protocolVersionIsValid(json)) {
      sendAck(commandId, false, "protocol_version_mismatch");
      return;
    }
    if (messageHasStringField(json, "mode", "tracking")) {
      setModeIndicators("tracking");
    } else if (messageHasStringField(json, "mode", "landing")) {
      setModeIndicators("landing");
    } else if (messageHasStringField(json, "mode", "idle")) {
      setModeIndicators("idle");
    } else {
      sendAck(commandId, false, "invalid_mode");
      return;
    }
    sendAck(commandId, true, "mode_set");
    return;
  }

  if (
    !messageHasType(json, "servo_target")
    && !messageHasType(json, "neutral")
  ) {
    sendAck(commandId, false, "unsupported_command");
    return;
  }
  if (!protocolVersionIsValid(json)) {
    sendAck(commandId, false, "protocol_version_mismatch");
    return;
  }

  float requestedAngleDeg = SERVO_NEUTRAL_ANGLE_DEG;
  if (!readFloatField(json, "angle_deg", &requestedAngleDeg)) {
    sendAck(commandId, false, "missing_angle_deg");
    return;
  }
  const float boundedAngleDeg = clampFloat(
    requestedAngleDeg,
    SERVO_MIN_ANGLE_DEG,
    SERVO_MAX_ANGLE_DEG
  );
  writeServoAngle(boundedAngleDeg);
  sendAck(
    commandId,
    true,
    boundedAngleDeg == requestedAngleDeg ? "ok" : "angle_clamped"
  );
}

static void pollSerialCommands() {
  while (Serial.available() > 0) {
    const char incoming = (char)Serial.read();
    if (incoming == '\r') continue;
    if (incoming == '\n') {
      if (commandLength > 0) {
        commandBuffer[commandLength] = '\0';
        processCommand(commandBuffer);
        commandLength = 0;
      }
      continue;
    }
    if (commandLength < MAX_COMMAND_LENGTH - 1) {
      commandBuffer[commandLength++] = incoming;
    } else {
      commandLength = 0;
      sendAck(-1, false, "command_too_long");
    }
  }
}

static bool startTof() {
  if (!tof.begin()) return false;
  tof.startRangeContinuous();
  return true;
}

static void sendTofSample(
  unsigned long nowMs,
  uint16_t distanceMm,
  uint8_t rangeStatus,
  bool valid
) {
  Serial.printf(
    "{\"type\":\"tof\",\"device_time_ms\":%lu,\"distance_mm\":%u,"
    "\"valid\":%s,\"range_status\":%u}\n",
    nowMs,
    distanceMm,
    valid ? "true" : "false",
    rangeStatus
  );
}

static void pollTof(unsigned long nowMs) {
  if (!tofReady) {
    if (nowMs - lastSensorAttemptMs >= SENSOR_RETRY_INTERVAL_MS) {
      lastSensorAttemptMs = nowMs;
      tofReady = startTof();
    }
    if (nowMs - lastInvalidSampleMs >= INVALID_SAMPLE_INTERVAL_MS) {
      lastInvalidSampleMs = nowMs;
      sendTofSample(nowMs, 0, 255, false);
    }
    return;
  }
  if (!tof.isRangeComplete()) return;

  const uint16_t distanceMm = tof.readRange();
  const uint8_t rangeStatus = tof.readRangeStatus();
  const bool valid = (
    distanceMm > 0
    && distanceMm < 8190
    && rangeStatus != 4
  );
  sendTofSample(nowMs, distanceMm, rangeStatus, valid);
}

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);

  trackingIndicator.begin();
  trackingIndicator.setBrightness(PREACT_LED_BRIGHTNESS);
  landingIndicator.begin();
  landingIndicator.setBrightness(PREACT_LED_BRIGHTNESS);
  setModeIndicators("idle");

  pinMode(PREACT_SERVO_PIN, OUTPUT);
  analogWriteFrequency(PREACT_SERVO_PIN, SERVO_FREQUENCY_HZ);
  analogWriteResolution(PREACT_SERVO_PIN, SERVO_PWM_BITS);
  writeServoAngle(SERVO_NEUTRAL_ANGLE_DEG);

  tofReady = startTof();
  lastSensorAttemptMs = millis();

  Serial.printf(
    "{\"type\":\"hello\",\"device\":\"genesis-mini\","
    "\"protocol_version\":%d,\"firmware_version\":\"%s\"}\n",
    PREACT_PROTOCOL_VERSION,
    PREACT_FIRMWARE_VERSION
  );
}

void loop() {
  const unsigned long nowMs = millis();
  pollSerialCommands();
  pollTof(nowMs);
  delay(1);
}
