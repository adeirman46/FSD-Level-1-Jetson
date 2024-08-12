#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

// Pin Definitions
#define SPEED_PWM_PIN 5
#define BRAKE_RPWM 25
#define BRAKE_LPWM 26
#define BRAKE_R_EN 27
#define BRAKE_L_EN 14
#define POT_PIN 34
#define PEDAL_PIN 2

#define MAX_BUFF_LEN 255

// PWM Configuration
#define PWM_FREQUENCY 5000
#define PWM_RESOLUTION 8

// Variables for serial communication
char c;
char s[MAX_BUFF_LEN];
uint8_t idx = 0;

// Variables for the Speed PID controller
volatile float desired_velocity = 0;
volatile float actual_velocity = 0;
volatile float speed_u = 0;

// PID coefficients for velocity
float speed_kp = 1.0;
float speed_ki = 0.001;
float speed_kd = 0.001;

// Speed PID state variables
float speed_previous_error = 0;
float speed_integral = 0;

// Variables for the Brake PID controller
volatile float desired_brake = 0;
volatile float actual_brake = 0;
volatile float brake_u = 0;

// PID coefficients for brake
float brake_kp = 4.0;
float brake_ki = 0.003;
float brake_kd = 0.0;

// Brake PID state variables
float brake_previous_error = 0;
float brake_integral = 0;

volatile int potValue = 0;
volatile int brake_state = 0;

// FreeRTOS handles
SemaphoreHandle_t xMutex = NULL;
QueueHandle_t xSerialQueue = NULL;

void readSerialData(void *parameter) {
  while (1) {
    if (Serial.available() > 0) {
      c = Serial.read();
      if (c != '\n' && idx < MAX_BUFF_LEN - 1) {
        s[idx++] = c;
      } else {
        s[idx] = '\0';
        idx = 0;

        char *token = strtok(s, ",");
        if (token != NULL) {
          float new_desired_velocity = atof(token);
          token = strtok(NULL, ",");
          if (token != NULL) {
            float new_desired_brake = atof(token);
            token = strtok(NULL, ",");
            if (token != NULL) {
              float new_actual_velocity = atof(token);
              token = strtok(NULL, ",");
              if (token != NULL) {
                int new_brake_state = atoi(token);
                
                xSemaphoreTake(xMutex, portMAX_DELAY);
                desired_velocity = new_desired_velocity;
                desired_brake = new_desired_brake;
                actual_velocity = new_actual_velocity;
                brake_state = new_brake_state;
                xSemaphoreGive(xMutex);
              }
            }
          }
        }
      }
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

void computeSpeedPID() {
  float error = desired_velocity - actual_velocity;
  float p = speed_kp * error;
  speed_integral += error;
  speed_integral = constrain(speed_integral, -154 / speed_ki, 154 / speed_ki);
  float i = speed_ki * speed_integral;
  float derivative = error - speed_previous_error;
  float d = speed_kd * derivative;
  speed_u = p + i + d;
  speed_u = constrain(speed_u, 0, 154);
  speed_previous_error = error;
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
  brake_previous_error = error;
}

void applyBrake() {
  if (brake_state == 1 && potValue >= 1500) {
    int brakeValue = map(potValue, 880, 2680, 0, 255);
    ledcWrite(1, brakeValue);
    ledcWrite(2, 0);
  }
  else if (brake_state == 0 && brake_u >= 0) {
    ledcWrite(1, brake_u);
    ledcWrite(2, 0);
  }
  else if (brake_state == 0 && brake_u < 0) {
    ledcWrite(1, 0);
    ledcWrite(2, abs(brake_u));
  }
}

void controlTask(void *parameter) {
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(1);
  xLastWakeTime = xTaskGetTickCount();
  
  while (1) {
    xSemaphoreTake(xMutex, portMAX_DELAY);
    potValue = analogRead(POT_PIN);
    actual_brake = potValue;
    
    computeSpeedPID();
    computeBrakePID();
    
    ledcWrite(0, (int)speed_u);
    applyBrake();
    xSemaphoreGive(xMutex);

    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

void serialOutputTask(void *parameter) {
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(1);
  xLastWakeTime = xTaskGetTickCount();
  
  while (1) {
    xSemaphoreTake(xMutex, portMAX_DELAY);
    float local_actual_brake = actual_brake;
    int local_brake_state = brake_state;
    xSemaphoreGive(xMutex);

    Serial.print("ESP: ");
    Serial.print("Desired Velocity: ");
    Serial.print(desired_velocity);
    Serial.print(", Desired Brake: ");
    Serial.print(desired_brake);
    Serial.print(", Actual Velocity: ");
    Serial.print(actual_velocity);
    Serial.print(", Actual Brake: ");
    Serial.print(local_actual_brake);
    Serial.print(", Speed u: ");
    Serial.print(speed_u);
    Serial.print(", Brake u: ");
    Serial.print(brake_u);
    Serial.print(", Pot Value: ");
    Serial.print(potValue);
    // Serial.print(", Brake State: ");
    // Serial.println(local_brake_state);

    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

void setup() {
  Serial.begin(115200);
  
  ledcSetup(0, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcAttachPin(SPEED_PWM_PIN, 0);
  
  ledcSetup(1, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcAttachPin(BRAKE_RPWM, 1);
  ledcSetup(2, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcAttachPin(BRAKE_LPWM, 2);
  
  pinMode(BRAKE_R_EN, OUTPUT);
  pinMode(BRAKE_L_EN, OUTPUT);
  digitalWrite(BRAKE_R_EN, HIGH);
  digitalWrite(BRAKE_L_EN, HIGH);
  
  pinMode(POT_PIN, INPUT);

  xMutex = xSemaphoreCreateMutex();

  xTaskCreate(readSerialData, "SerialTask", 2048, NULL, 1, NULL);
  xTaskCreate(controlTask, "ControlTask", 2048, NULL, 2, NULL);
  xTaskCreate(serialOutputTask, "OutputTask", 2048, NULL, 1, NULL);
}

void loop() {
  // Empty. Tasks are handled by FreeRTOS
}