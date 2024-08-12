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

#define MAX_BUFF_LEN 255

// PWM Configuration
#define PWM_FREQUENCY 5000
#define PWM_RESOLUTION 8

// Variables for serial communication
char c;
char s[MAX_BUFF_LEN];
uint8_t idx = 0;

// Variables for the MPC controller
volatile float desired_velocity = 0;
volatile float actual_velocity = 0;
volatile float speed_u = 0;

volatile float desired_brake = 0;
volatile float actual_brake = 0;
volatile float brake_u = 0;

volatile int potValue = 0;

// MPC parameters
const int HORIZON = 5;
const int STATE_DIM = 2;  // velocity and brake position
const int INPUT_DIM = 2;  // speed input and brake input

// Simplified model matrices (these need to be tuned for your system)
float A[STATE_DIM][STATE_DIM] = {{1.0, 0.0}, {0.0, 1.0}};
float B[STATE_DIM][INPUT_DIM] = {{0.1, 0.0}, {0.0, 0.1}};

// Cost matrices
float Q[STATE_DIM] = {1.0, 1.0};  // State cost
float R[INPUT_DIM] = {0.1, 0.1};  // Input cost

// FreeRTOS handles
SemaphoreHandle_t xMutex = NULL;

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
            
            xSemaphoreTake(xMutex, portMAX_DELAY);
            desired_velocity = new_desired_velocity;
            desired_brake = new_desired_brake;
            xSemaphoreGive(xMutex);
          }
        }
      }
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

void simpleMPC(float current_state[STATE_DIM], float target_state[STATE_DIM], float control_output[INPUT_DIM]) {
  float state[STATE_DIM];
  float next_state[STATE_DIM];
  float best_cost = 1e10;
  float best_input[INPUT_DIM] = {0};

  // Simple grid search for optimal input
  for (float u1 = -1; u1 <= 1; u1 += 0.1) {
    for (float u2 = -1; u2 <= 1; u2 += 0.1) {
      float total_cost = 0;
      state[0] = current_state[0];
      state[1] = current_state[1];

      for (int i = 0; i < HORIZON; i++) {
        // Compute next state
        next_state[0] = A[0][0] * state[0] + A[0][1] * state[1] + B[0][0] * u1 + B[0][1] * u2;
        next_state[1] = A[1][0] * state[0] + A[1][1] * state[1] + B[1][0] * u1 + B[1][1] * u2;

        // Compute cost
        float state_cost = Q[0] * pow(next_state[0] - target_state[0], 2) + 
                           Q[1] * pow(next_state[1] - target_state[1], 2);
        float input_cost = R[0] * u1 * u1 + R[1] * u2 * u2;
        total_cost += state_cost + input_cost;

        // Update state for next iteration
        state[0] = next_state[0];
        state[1] = next_state[1];
      }

      if (total_cost < best_cost) {
        best_cost = total_cost;
        best_input[0] = u1;
        best_input[1] = u2;
      }
    }
  }

  control_output[0] = best_input[0];
  control_output[1] = best_input[1];
}

void controlTask(void *parameter) {
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(1);  // Run every 10ms
  xLastWakeTime = xTaskGetTickCount();
  
  float current_state[STATE_DIM];
  float target_state[STATE_DIM];
  float control_output[INPUT_DIM];
  
  while (1) {
    xSemaphoreTake(xMutex, portMAX_DELAY);
    potValue = analogRead(POT_PIN);
    actual_brake = map(potValue, 0, 1023, 0, 100);  // Map pot value to 0-100 range
    
    current_state[0] = actual_velocity;
    current_state[1] = actual_brake;
    target_state[0] = desired_velocity;
    target_state[1] = desired_brake;
    
    simpleMPC(current_state, target_state, control_output);
    
    speed_u = control_output[0];
    brake_u = control_output[1];
    
    // Apply control actions
    ledcWrite(0, (int)constrain(map(speed_u, -1, 1, 0, 255), 0, 154));
    if (brake_u >= 0) {
      ledcWrite(1, (int)constrain(map(brake_u, 0, 1, 0, 255), 0, 255));
      ledcWrite(2, 0);
    } else {
      ledcWrite(1, 0);
      ledcWrite(2, (int)constrain(map(-brake_u, 0, 1, 0, 255), 0, 255));
    }
    
    xSemaphoreGive(xMutex);

    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

void serialOutputTask(void *parameter) {
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(1);  // Run every 100ms
  xLastWakeTime = xTaskGetTickCount();
  
  while (1) {
    xSemaphoreTake(xMutex, portMAX_DELAY);
    float local_actual_brake = actual_brake;
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
    Serial.println(brake_u);

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
  xTaskCreate(controlTask, "ControlTask", 4096, NULL, 2, NULL);
  xTaskCreate(serialOutputTask, "OutputTask", 2048, NULL, 1, NULL);
}

void loop() {
  // Empty. Tasks are handled by FreeRTOS
}