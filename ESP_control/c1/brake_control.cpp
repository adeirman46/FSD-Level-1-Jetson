#include <Arduino.h>
#define RPWM 10
#define LPWM 11
#define R_EN 8
#define L_EN 9
#define potpin A0

double p = 4, i = 0.003, d = 0;
double input = 0, consig = 0;
double setpoint=0;

double prevError = 0;
double integral = 0;
unsigned long lastTime = 0;
double desiredAngle = 270;
double currentAngle = 0;

int potValue = 0;

void setup() {
  // Initialize Serial communication
  Serial.begin(9600);
  
  // Setup motor driver pins
  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(R_EN, OUTPUT);
  pinMode(L_EN, OUTPUT);
  
  digitalWrite(R_EN, HIGH); // Enable clockwise rotation
  digitalWrite(L_EN, HIGH); // Enable counterclockwise rotation
}

void loop() {

  potValue = analogRead(potpin);
  input = potValue; // Use potValue as the input for PID calculation

  if (Serial.available() > 0) {
    desiredAngle = Serial.parseFloat();
    if(desiredAngle != NULL){
      setpoint = desiredAngle;
    }
  }
  
  double error = setpoint - input;
  integral += error;
  double derivative = error - prevError;

  consig = p * error + i * integral + d * derivative;
  prevError = error;

  if (consig < 0) {
    analogWrite(RPWM, constrain(-consig, 0, 255)); // Clockwise rotation
    analogWrite(LPWM, 0);
  } else {
    analogWrite(RPWM, 0);
    analogWrite(LPWM, constrain(consig, 0, 255)); // Counterclockwise rotation
  }
  Serial.print("Desire : ");
  Serial.print(desiredAngle);
  Serial.print(" Set: ");
  Serial.print(setpoint);
  Serial.print(" CurrentPos: ");
  Serial.print(input);
  Serial.print(" Controlsig: ");
  Serial.println(consig);
  delay(1);
}