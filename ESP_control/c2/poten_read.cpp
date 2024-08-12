#include <Arduino.h>
#define PIN_POT 25

void setup(){
    Serial.begin(115200);
    pinMode(PIN_POT, OUTPUT);
}

void loop(){
    analogReadResolution(10);
    // analogSetAttenuation(ADC_6db);
    int data = analogRead(PIN_POT);
    double x = data * 3.3 / 4095 * 100;
    Serial.print("Potentiometer Value: ");
    Serial.println(x);
    delay(100);
}