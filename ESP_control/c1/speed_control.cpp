#include <Arduino.h>

#define MAX_BUFF_LEN 255

char c;
char s[MAX_BUFF_LEN];
uint8_t idx = 0;

// Variables for the PID controller
float desired_velocity = 0;
float desired_brake = 0;
float actual_velocity = 0;

float u = 0;  // Control variable

// PID coefficients for velocity
float kp = 4.0;  // Proportional gain
float ki = 0.007;  // Integral gain
float kd = 0.0; // Derivative gain

// PID state variables
float previous_error = 0;
float integral = 0;

void readSerialData() {
  // Check for new serial data
  while (Serial.available() > 0) {
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
}

void computePID() {
  // Calculate error
  float error = desired_velocity - actual_velocity;

  // Proportional term
  float p = kp * error;

  // Integral term
  integral += error;
  float i = ki * integral;

  // Derivative term
  float derivative = error - previous_error;
  float d = kd * derivative;

  // Calculate control variable
  u = p + i + d;

  // Clamp the control variable to acceptable limits (e.g., PWM range)
  u = constrain(u, 0, 154);

  // Update previous error
  previous_error = error;
}

void printDebugInfo() {
  Serial.print("ESP: ");
  Serial.print("Desired Velocity: ");
  Serial.print(desired_velocity);
  Serial.print(", Desired Brake: ");
  Serial.print(desired_brake);
  Serial.print(", Actual Velocity: ");
  Serial.print(actual_velocity);
  Serial.print(", u: ");
  Serial.println(u);
}

void setup() {
  Serial.begin(115200);
  pinMode(5, OUTPUT);
}

void loop() {
  // Read and process serial data
  readSerialData();

  // Compute PID
  computePID();

  // Write to the actuator
  analogWrite(5, (int)u);

  // Print debug information
  printDebugInfo();

  // Small delay to simulate control loop timing
  delay(1);
}
