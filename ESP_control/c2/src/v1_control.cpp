#include <Arduino.h>

#define MAX_BUFF_LEN 255

// Pin Definitions
#define SPEED_PWM_PIN 5
#define BRAKE_RPWM 25
#define BRAKE_LPWM 26
#define BRAKE_R_EN 27
#define BRAKE_L_EN 14
#define POT_PIN 34
#define PEDAL_PIN 2

char c;
char s[MAX_BUFF_LEN];
uint8_t idx = 0;

// Variables for the PID controller
float desired_velocity = 0;
float desired_brake = 0;
float actual_velocity = 0;
float actual_brake = 0;

float u = 0;  // Control variable

// PID coefficients for velocity
float kp = 10.0;  // Proportional gain
//float ki = 0.01;  // Integral gain
float ki = 0.5;
float kd = 0.0; // Derivative gain

// PID state variables
float previous_error = 0;
float integral = 0;

// PID coefficients for brake
float brake_kp = 4.0;
float brake_ki = 0.003;
float brake_kd = 0.0;

// Brake PID state variables
float brake_previous_error = 0;
float brake_integral = 0;

volatile int potValue = 0;
volatile int pedalValue = 0;
volatile float brake_u = 0;


//moving average
const int numReadings = 10;  // Number of points for moving average
int readings[numReadings];  // The readings from the analog input
int readIndex = 0;          // The index of the current reading
int total = 0;              // The running total
int average = 0;  

void setup() {
  Serial.begin(115200);
  pinMode(SPEED_PWM_PIN, OUTPUT);
  pinMode(BRAKE_R_EN, OUTPUT);
  pinMode(BRAKE_L_EN, OUTPUT);

  digitalWrite(BRAKE_R_EN, HIGH);
  digitalWrite(BRAKE_L_EN, HIGH);

  pinMode(POT_PIN, INPUT);
  pinMode(PEDAL_PIN, OUTPUT);

  // Configure PWM channels for ESP32
  ledcAttachPin(BRAKE_RPWM, 0);
  ledcAttachPin(BRAKE_RPWM, 1);
  ledcAttachPin(SPEED_PWM_PIN, 2);
  ledcSetup(0, 5000, 8); // 5 kHz PWM, 8-bit resolution
  ledcSetup(1, 5000, 8); // 5 kHz PWM, 8-bit resolution
  ledcSetup(2, 5000, 8); // 5 kHz PWM, 8-bit resolution

  //for potensio reading
  // Initialize all readings to 0
  for (int i = 0; i < numReadings; i++) {
    readings[i] = 0;
  }


}

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

void computeBrakePID() {
  float error = desired_brake - actual_brake;
  float p = brake_kp * error;
  brake_integral += error;
  float i = brake_ki * brake_integral;
  float derivative = error - brake_previous_error;
  float d = brake_kd * derivative;
  brake_u = p + i + d;
  brake_u = constrain(brake_u, 0, 255);
  if(error<5){
    brake_u = 0;
  }
  brake_previous_error = error;
}

void printDebugInfo() {
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
  Serial.println(u);
}

void loop() {
  // Read and process serial data
  readSerialData();

  //potensio reading
  analogReadResolution(10);
  total = total - readings[readIndex];
  readings[readIndex]  = analogRead(POT_PIN);
  total = total + readings[readIndex];
  readIndex = readIndex + 1;
  if (readIndex >= numReadings) {
    readIndex = 0;
  }
  actual_brake = total / numReadings;


  //pedal
  pedalValue = analogRead(PEDAL_PIN);

  // Compute PID
  computePID();
  computeBrakePID();

  

  if (pedalValue >= 375){
    int mappedValue = map(pedalValue, 0, 1023, 0, 255);
    analogWrite(SPEED_PWM_PIN, (int)mappedValue);
  }
  else{
    // Write to the actuator
    ledcWrite(2, (int)u);
    // analogWrite(5, (int)u);
  }
  
  if (brake_u > 0) {
    ledcWrite(0, (int)brake_u);
    ledcWrite(1, 0);
  } else {
    ledcWrite(0, 0);
    ledcWrite(1, (int)-brake_u);
  }

  // Print debug information
  printDebugInfo();

  // Small delay to simulate control loop timing
  delay(1);
}