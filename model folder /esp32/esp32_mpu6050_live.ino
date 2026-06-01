#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define MPU_ADDR 0x68

const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
// Replace with your laptop LAN IP from: ipconfig getifaddr en0
const char* API_URL   = "http://192.168.1.23:5000/api/live-sample";

LiquidCrystal_I2C lcd(0x27, 16, 2);

unsigned long lastSendMs = 0;
const unsigned long SEND_INTERVAL_MS = 40; // 25 Hz

void mpuWrite(uint8_t reg, uint8_t data) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission();
}

int16_t read16(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)2, (uint8_t)true);
  int16_t hi = Wire.read();
  int16_t lo = Wire.read();
  return (hi << 8) | lo;
}

void readMPU(int16_t &ax, int16_t &ay, int16_t &az, int16_t &gx, int16_t &gy, int16_t &gz) {
  ax = read16(0x3B);
  ay = read16(0x3D);
  az = read16(0x3F);
  gx = read16(0x43);
  gy = read16(0x45);
  gz = read16(0x47);
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void setupMPU() {
  Wire.begin();
  delay(50);
  mpuWrite(0x6B, 0x00); // wake
  mpuWrite(0x1C, 0x00); // accel +-2g
  mpuWrite(0x1B, 0x00); // gyro +-250 dps
  delay(50);
}

void setupLCD() {
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("MPU6050 READY");
}

void postSample(float xAcc, float yAcc, float zAcc, float xGyro, float yGyro, float zGyro) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(API_URL);
  http.addHeader("Content-Type", "application/json");

  String body = "{";
  body += "\"xAcc\":" + String(xAcc, 6) + ",";
  body += "\"yAcc\":" + String(yAcc, 6) + ",";
  body += "\"zAcc\":" + String(zAcc, 6) + ",";
  body += "\"xGyro\":" + String(xGyro, 6) + ",";
  body += "\"yGyro\":" + String(yGyro, 6) + ",";
  body += "\"zGyro\":" + String(zGyro, 6);
  body += "}";

  int code = http.POST(body);
  if (code > 0) {
    String resp = http.getString();
    Serial.print("POST ");
    Serial.print(code);
    Serial.print(" -> ");
    Serial.println(resp);
  } else {
    Serial.print("HTTP error: ");
    Serial.println(code);
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  setupMPU();
  setupLCD();
  connectWiFi();
}

void loop() {
  int16_t axRaw, ayRaw, azRaw, gxRaw, gyRaw, gzRaw;
  readMPU(axRaw, ayRaw, azRaw, gxRaw, gyRaw, gzRaw);

  // convert to physical units
  float xAcc = axRaw / 16384.0f;
  float yAcc = ayRaw / 16384.0f;
  float zAcc = azRaw / 16384.0f;
  float xGyro = gxRaw / 131.0f;
  float yGyro = gyRaw / 131.0f;
  float zGyro = gzRaw / 131.0f;

  // serial output (all 6)
  Serial.print("xAcc:"); Serial.print(xAcc, 3);
  Serial.print(" yAcc:"); Serial.print(yAcc, 3);
  Serial.print(" zAcc:"); Serial.print(zAcc, 3);
  Serial.print(" xGyro:"); Serial.print(xGyro, 3);
  Serial.print(" yGyro:"); Serial.print(yGyro, 3);
  Serial.print(" zGyro:"); Serial.println(zGyro, 3);

  // lcd (3 values visible)
  lcd.setCursor(0, 0);
  lcd.print("Ax:"); lcd.print(xAcc, 2);
  lcd.print(" Ay:"); lcd.print(yAcc, 2);

  lcd.setCursor(0, 1);
  lcd.print("Az:"); lcd.print(zAcc, 2);
  lcd.print("      ");

  unsigned long now = millis();
  if (now - lastSendMs >= SEND_INTERVAL_MS) {
    lastSendMs = now;
    postSample(xAcc, yAcc, zAcc, xGyro, yGyro, zGyro);
  }

  delay(10);
}
