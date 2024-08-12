#include <Arduino.h>

#define RPWM 25
#define LPWM 26
#define R_EN 27
#define L_EN 14
#define potpin 34

#define SPEED_PWM_PIN 5
#define PEDAL_PIN 2

// brake
double brake_kp = 5, brake_ki = 0.05, brake_kd = 0;

// pedal
double pedal_kp = 1, pedal_ki = 0.01, pedal_kd = 0;

double input = 0, consig = 0;
double input_v = 0, consig_v = 0;

double setpoint=0;
double setpoint_v = 0;

double prevError = 0;
double prevError_v = 0;

double integral = 0;
double integral_v = 0;

unsigned long lastTime = 0;
unsigned long lastTime_v = 0;

double desiredAngle = 270;
double currentAngle = 0;

int potValue = 0;

#define WINDOW_SIZE 10

int readings[WINDOW_SIZE];
int readIndex = 0;
int total = 0;
int average = 0;


#define MAX_BUFF_LEN 255
char c;
char s[MAX_BUFF_LEN];
uint8_t idx = 0;


// Variables for the PID controller
float desired_velocity = 0;
float desired_brake = 0;
float actual_velocity = 0;
float actual_brake = 0;

volatile int pedalValue = 0;


void setup() {
  // Initialize Serial communication
  Serial.begin(115200);
  
  // Setup motor driver pins
  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(R_EN, OUTPUT);
  pinMode(L_EN, OUTPUT);

  pinMode(potpin, INPUT);
  pinMode(PEDAL_PIN, INPUT);
  pinMode(SPEED_PWM_PIN, OUTPUT);
  
  digitalWrite(R_EN, HIGH); // Enable clockwise rotation
  digitalWrite(L_EN, HIGH); // Enable counterclockwise rotation

  // Configure PWM channels for ESP32
  ledcAttachPin(RPWM, 0);
  ledcAttachPin(LPWM, 1);
  ledcAttachPin(SPEED_PWM_PIN, 2);

  ledcSetup(0, 5000, 8); // 5 kHz PWM, 8-bit resolution
  ledcSetup(1, 5000, 8); // 5 kHz PWM, 8-bit resolution
  ledcSetup(2, 5000, 8); // 5 kHz PWM, 8-bit resolution

  lastTime = millis();
  lastTime_v = millis();
}

void loop() {

  analogReadResolution(10);
  potValue = analogRead(potpin);

  pedalValue = analogRead(PEDAL_PIN);

  // // Subtract the last reading
  total = total - readings[readIndex];
  // Read from the sensor
  readings[readIndex] = analogRead(potpin);
  // Add the reading to the total
  total = total + readings[readIndex];
  // Advance to the next position in the array
  readIndex = (readIndex + 1) % WINDOW_SIZE;

  // Calculate the average
  average = total / WINDOW_SIZE;

  // Use 'average' instead of 'potValue' in your PID calculations
  input = potValue;
  // input = potValue; // Use potValue as the input for PID calculation

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

  unsigned long currentTime = millis();
  double elapsedTime = (currentTime - lastTime) / 1000.0; // convert to seconds
  lastTime = currentTime;
  
  setpoint = desired_brake;
  double error = setpoint - input;
  integral += error * elapsedTime;
  double derivative = (error - prevError) / elapsedTime;

  consig = brake_kp * error + brake_ki * integral + brake_kd * derivative;
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

  unsigned long currentTime_v = millis();
  double elapsedTime_v = (currentTime_v - lastTime_v) / 1000.0; // convert to seconds
  lastTime_v = currentTime_v;

  input_v = desired_velocity;
  double error_v = setpoint_v - input_v;
  integral_v += error_v * elapsedTime_v;
  double derivative_v = (error_v - prevError_v) / elapsedTime_v;

  consig_v = pedal_kp * error_v + pedal_ki * integral_v + pedal_kd * derivative_v;
  prevError_v = error_v;

  // Anti-windup for integral term
  if (consig_v > 154) {
    consig_v = 154;
    integral_v -= error_v * elapsedTime_v; // Prevent further increase in integral
  } 
  else if (consig_v < 0) {
    consig_v = 0;
    integral_v -= error_v * elapsedTime_v; // Prevent further decrease in integral
  }

  if (pedalValue >= 375){
    int mappedValue = map(pedalValue, 0, 1023, 0, 255);
    ledcWrite(2, (int)mappedValue);
  }
  else{
    // Write to the actuator
    ledcWrite(2, (int)consig_v);
    // analogWrite(5, (int)u);
  }

  Serial.print("ESP: ");
  Serial.print("Desired Velocity: ");
  Serial.print(desired_velocity);
  Serial.print(", Desired Brake: ");
  Serial.print(desired_brake);
  Serial.print(", Actual Velocity: ");
  Serial.print(actual_velocity);
  Serial.print(", Actual Brake: ");
  Serial.print(actual_brake);
  Serial.print(", u: ");
  Serial.println(consig_v);

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