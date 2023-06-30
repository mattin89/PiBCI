#include <BluetoothSerial.h>

BluetoothSerial SerialBT;

const int numPins = 8;
const int adcPins[numPins] = {26, 25, 34, 39, 36, 4, 13, 12};  // Example ADC pins

int pinValues[numPins];

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32"); // Bluetooth device name

  for (int i = 0; i < numPins; i++) {
    pinMode(adcPins[i], INPUT);
  }
}

void loop() {
  for (int i = 0; i < numPins; i++) {
    pinValues[i] = analogRead(adcPins[i]);
    Serial.print("Pin ");
    Serial.print(adcPins[i]);
    Serial.print(": ");
    Serial.println(pinValues[i]);
  }

  SerialBT.write((uint8_t*)pinValues, sizeof(pinValues));

  delay(1000); // Delay between transmissions
}