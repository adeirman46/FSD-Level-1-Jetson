#include <Arduino.h>

#define RPWM 25
#define LPWM 26
#define R_EN 27
#define L_EN 14
#define potpin 34

// double p = 4, i = 0.003, d = 0;
double p = 5, i = 0.05, d = 0;

double input = 0, consig = 0;
double setpoint=0;

double prevError = 0;
double integral = 0;
unsigned long lastTime = 0;
double desiredAngle = 270;
double currentAngle = 0;

int potValue = 0;

#define WINDOW_SIZE 10

int readings[WINDOW_SIZE];
int readIndex = 0;
int total = 0;
int average = 0;

void setup() {
  // Initialize Serial communication
  Serial.begin(115200);
  
  // Setup motor driver pins
  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(R_EN, OUTPUT);
  pinMode(L_EN, OUTPUT);
  
  digitalWrite(R_EN, HIGH); // Enable clockwise rotation
  digitalWrite(L_EN, HIGH); // Enable counterclockwise rotation

  // Configure PWM channels for ESP32
  ledcAttachPin(RPWM, 0);
  ledcAttachPin(LPWM, 1);
  ledcSetup(0, 5000, 8); // 5 kHz PWM, 8-bit resolution
  ledcSetup(1, 5000, 8); // 5 kHz PWM, 8-bit resolution

  lastTime = millis();
}

void loop() {

  // analogReadResolution(10);
  potValue = analogRead(potpin);

  // // // Subtract the last reading
  // total = total - readings[readIndex];
  // // Read from the sensor
  // readings[readIndex] = analogRead(potpin);
  // // Add the reading to the total
  // total = total + readings[readIndex];
  // // Advance to the next position in the array
  // readIndex = (readIndex + 1) % WINDOW_SIZE;

  // // Calculate the average
  // average = total / WINDOW_SIZE;

  // Use 'average' instead of 'potValue' in your PID calculations
  input = potValue;
  // input = potValue; // Use potValue as the input for PID calculation

  if (Serial.available() > 0) {
    desiredAngle = Serial.parseFloat();
    if(desiredAngle != 0){
      setpoint = desiredAngle;
    }
    // Serial.print("Received desired angle: ");
    // Serial.println(desiredAngle);
  }

  unsigned long currentTime = millis();
  double elapsedTime = (currentTime - lastTime) / 1000.0; // convert to seconds
  lastTime = currentTime;

  double error = setpoint - input;
  integral += error * elapsedTime;
  double derivative = (error - prevError) / elapsedTime;

  consig = p * error + i * integral + d * derivative;
  prevError = error;

  // Anti-windup for integral term
  if (consig > 255) {
    consig = 255;
    integral -= error * elapsedTime; // Prevent further increase in integral
  } else if (consig < -255) {
    consig = -255;
    integral -= error * elapsedTime; // Prevent further decrease in integral
  }

  if (consig < 0) {
    ledcWrite(0, constrain(-consig, 0, 255)); // Clockwise rotation
    ledcWrite(1, 0);
  } else {
    ledcWrite(0, 0);
    ledcWrite(1, constrain(consig, 0, 255)); // Counterclockwise rotation
  }
  
  Serial.print("Desire : ");
  Serial.print(desiredAngle);
  Serial.print(" Set: ");
  Serial.print(setpoint);
  Serial.print(" CurrentPos: ");
  Serial.print(input);
  Serial.print(" Controlsig: ");
  Serial.println(consig);
  
  delay(100);
}