#include <Arduino.h>

#define SPEED_PWM_PIN 5
#define BRAKE_RPWM 25
#define BRAKE_LPWM 26
#define BRAKE_R_EN 27
#define BRAKE_L_EN 14
#define POT_PIN 34
#define PEDAL_PIN 2

#define MAX_BUFF_LEN 255
#define WINDOW_SIZE 10
char c;
char s[MAX_BUFF_LEN];
uint8_t idx = 0;

double p = 5, i = 0.05, d = 0;
double brake_kp = 4.0, brake_ki = 0.003, brake_kd = 0.0;

// Variables for the PID controller
float desired_velocity = 0;
float desired_brake = 0;
float actual_velocity = 0;
float actual_brake = 0;

double prevError = 0;
double integral = 0;
double brake_prevError = 0;
double brake_integral = 0;

unsigned long lastTime = 0;
unsigned long brake_lastTime = 0;
double desiredAngle = 270;
double desiredBrake = 0;

int potValue = 0;
int pedalValue = 0;

int readings[WINDOW_SIZE];
int readIndex = 0;
int total = 0;
int average = 0;

void setup() {
  Serial.begin(115200);
  
  pinMode(SPEED_PWM_PIN, OUTPUT);
  pinMode(BRAKE_RPWM, OUTPUT);
  pinMode(BRAKE_LPWM, OUTPUT);
  pinMode(BRAKE_R_EN, OUTPUT);
  pinMode(BRAKE_L_EN, OUTPUT);
  pinMode(POT_PIN, INPUT);
  pinMode(PEDAL_PIN, INPUT);
  
  digitalWrite(BRAKE_R_EN, HIGH);
  digitalWrite(BRAKE_L_EN, HIGH);

  ledcAttachPin(SPEED_PWM_PIN, 0);
  ledcAttachPin(BRAKE_RPWM, 1);
  ledcAttachPin(BRAKE_LPWM, 2);
  ledcSetup(0, 5000, 8);
  ledcSetup(1, 5000, 8);
  ledcSetup(2, 5000, 8);

  lastTime = millis();
  brake_lastTime = millis();

  for (int i = 0; i < WINDOW_SIZE; i++) {
    readings[i] = 0;
  }
}

void computePID() {
  unsigned long currentTime = millis();
  double elapsedTime = (currentTime - lastTime) / 1000.0;
  lastTime = currentTime;

  double error = setpoint - input;
  integral += error * elapsedTime;
  double derivative = (error - prevError) / elapsedTime;

  consig = p * error + i * integral + d * derivative;
  prevError = error;

  if (consig > 154) {
    consig = 154;
    integral -= error * elapsedTime;
  } else if (consig < 0) {
    consig = 0;
    integral -= error * elapsedTime;
  }
}

void computeBrakePID() {
  unsigned long currentTime = millis();
  double elapsedTime = (currentTime - brake_lastTime) / 1000.0;
  brake_lastTime = currentTime;

  double error = brake_setpoint - brake_input;
  brake_integral += error * elapsedTime;
  double derivative = (error - brake_prevError) / elapsedTime;

  brake_consig = brake_kp * error + brake_ki * brake_integral + brake_kd * derivative;
  brake_consig = constrain(brake_consig, 0, 255);
  
  if (error < 5) {
    brake_consig = 0;
  }
  
  brake_prevError = error;
}

void loop() {
  potValue = analogRead(POT_PIN);
  pedalValue = analogRead(PEDAL_PIN);

  total = total - readings[readIndex];
  readings[readIndex] = potValue;
  
  total = total + readings[readIndex];
  readIndex = (readIndex + 1) % WINDOW_SIZE;
  average = total / WINDOW_SIZE;

  input = average;
  brake_input = average;

  if (Serial.available() > 0) {
    c = Serial.read();
    if (c != '\n' && idx < MAX_BUFF_LEN - 1) {  // Ensure buffer does not overflow
      s[idx++] = c;
    } else {
      s[idx] = '\0';  // Null-terminate the string
      idx = 0;

      // Process the string to extract three integers
      char *token = strtok(s, ",");
      if (token != NULL) {
        desired_velocity = atof(token);
        token = strtok(NULL, ",");
        if (token != NULL) {
          desired_brake = atof(token);
          token = strtok(NULL, ",");
          if (token != NULL) {
            actual_velocity = atof(token);
          }
        }
      }
    }
  }

  computePID();
  computeBrakePID();

  if (pedalValue >= 375) {
    int mappedValue = map(pedalValue, 0, 1023, 0, 255);
    ledcWrite(0, mappedValue);
  } else {
    ledcWrite(0, (int)consig);
  }

  if (brake_consig > 0) {
    ledcWrite(1, (int)brake_consig);
    ledcWrite(2, 0);
  } else {
    ledcWrite(1, 0);
    ledcWrite(2, (int)-brake_consig);
  }

  Serial.print("Desired Angle: ");
  Serial.print(desiredAngle);
  Serial.print(" Desired Brake: ");
  Serial.print(desiredBrake);
  Serial.print(" Current Pos: ");
  Serial.print(input);
  Serial.print(" Speed Control: ");
  Serial.print(consig);
  Serial.print(" Brake Control: ");
  Serial.println(brake_consig);

  delay(10);
}