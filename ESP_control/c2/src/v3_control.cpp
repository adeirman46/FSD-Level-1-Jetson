#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// Pin Definitions
#define SPEED_PWM_PIN 5
#define BRAKE_RPWM 25
#define BRAKE_LPWM 26
#define BRAKE_R_EN 27
#define BRAKE_L_EN 14
#define POT_PIN 34
#define PEDAL_PIN 2

// PID Constants
#define SPEED_KP 5.0
#define SPEED_KI 0.05
#define SPEED_KD 0.0

#define BRAKE_KP 4.0
#define BRAKE_KI 0.003
#define BRAKE_KD 0.0

// Task Priorities
#define PRIORITY_SENSOR_READ 3
#define PRIORITY_PID_COMPUTE 2
#define PRIORITY_ACTUATOR_CONTROL 2
#define PRIORITY_SERIAL_HANDLE 1

// Struct for sensor data
struct SensorData {
  int potValue;
  int pedalValue;
};

// Struct for control signals
struct ControlSignals {
  double speedControl;
  double brakeControl;
};

// Struct for setpoints
struct Setpoints {
  double speedSetpoint;
  double brakeSetpoint;
};

// Queue handles
QueueHandle_t sensorQueue;
QueueHandle_t controlQueue;
QueueHandle_t setpointQueue;

// Task handles
TaskHandle_t sensorTaskHandle;
TaskHandle_t speedPIDTaskHandle;
TaskHandle_t brakePIDTaskHandle;
TaskHandle_t actuatorTaskHandle;
TaskHandle_t serialTaskHandle;

// Function prototypes
void sensorTask(void *pvParameters);
void speedPIDTask(void *pvParameters);
void brakePIDTask(void *pvParameters);
void actuatorTask(void *pvParameters);
void serialTask(void *pvParameters);

void setup() {
  // Initialize Serial
  Serial.begin(115200);

  // Initialize pins
  pinMode(SPEED_PWM_PIN, OUTPUT);
  pinMode(BRAKE_RPWM, OUTPUT);
  pinMode(BRAKE_LPWM, OUTPUT);
  pinMode(BRAKE_R_EN, OUTPUT);
  pinMode(BRAKE_L_EN, OUTPUT);
  pinMode(POT_PIN, INPUT);
  pinMode(PEDAL_PIN, INPUT);

  digitalWrite(BRAKE_R_EN, HIGH);
  digitalWrite(BRAKE_L_EN, HIGH);

  // Setup PWM channels
  ledcAttachPin(SPEED_PWM_PIN, 0);
  ledcAttachPin(BRAKE_RPWM, 1);
  ledcAttachPin(BRAKE_LPWM, 2);
  ledcSetup(0, 5000, 8);
  ledcSetup(1, 5000, 8);
  ledcSetup(2, 5000, 8);

  // Create queues
  sensorQueue = xQueueCreate(1, sizeof(SensorData));
  controlQueue = xQueueCreate(1, sizeof(ControlSignals));
  setpointQueue = xQueueCreate(1, sizeof(Setpoints));

  // Create tasks
  xTaskCreatePinnedToCore(sensorTask, "SensorTask", 2048, NULL, PRIORITY_SENSOR_READ, &sensorTaskHandle, 0);
  xTaskCreatePinnedToCore(speedPIDTask, "SpeedPIDTask", 2048, NULL, PRIORITY_PID_COMPUTE, &speedPIDTaskHandle, 1);
  xTaskCreatePinnedToCore(brakePIDTask, "BrakePIDTask", 2048, NULL, PRIORITY_PID_COMPUTE, &brakePIDTaskHandle, 1);
  xTaskCreatePinnedToCore(actuatorTask, "ActuatorTask", 2048, NULL, PRIORITY_ACTUATOR_CONTROL, &actuatorTaskHandle, 0);
  xTaskCreatePinnedToCore(serialTask, "SerialTask", 2048, NULL, PRIORITY_SERIAL_HANDLE, &serialTaskHandle, 1);
}

void loop() {
  // Empty. Things are now done in tasks.
}

void sensorTask(void *pvParameters) {
  SensorData sensorData;
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(10); // 100Hz

  xLastWakeTime = xTaskGetTickCount();

  for(;;) {
    sensorData.potValue = analogRead(POT_PIN);
    sensorData.pedalValue = analogRead(PEDAL_PIN);
    xQueueOverwrite(sensorQueue, &sensorData);
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

void speedPIDTask(void *pvParameters) {
  SensorData sensorData;
  Setpoints setpoints;
  ControlSignals controlSignals;
  double input, setpoint, error, lastError = 0, integral = 0;
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(20); // 50Hz

  xLastWakeTime = xTaskGetTickCount();

  for(;;) {
    if (xQueuePeek(sensorQueue, &sensorData, 0) == pdTRUE &&
        xQueuePeek(setpointQueue, &setpoints, 0) == pdTRUE) {
      
      input = sensorData.potValue;
      setpoint = setpoints.speedSetpoint;
      
      error = setpoint - input;
      integral += error;
      double derivative = error - lastError;

      controlSignals.speedControl = SPEED_KP * error + SPEED_KI * integral + SPEED_KD * derivative;
      controlSignals.speedControl = constrain(controlSignals.speedControl, 0, 154);

      xQueueOverwrite(controlQueue, &controlSignals);

      lastError = error;
    }
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

void brakePIDTask(void *pvParameters) {
  SensorData sensorData;
  Setpoints setpoints;
  ControlSignals controlSignals;
  double input, setpoint, error, lastError = 0, integral = 0;
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(20); // 50Hz

  xLastWakeTime = xTaskGetTickCount();

  for(;;) {
    if (xQueuePeek(sensorQueue, &sensorData, 0) == pdTRUE &&
        xQueuePeek(setpointQueue, &setpoints, 0) == pdTRUE &&
        xQueuePeek(controlQueue, &controlSignals, 0) == pdTRUE) {
      
      input = sensorData.potValue;
      setpoint = setpoints.brakeSetpoint;
      
      error = setpoint - input;
      integral += error;
      double derivative = error - lastError;

      controlSignals.brakeControl = BRAKE_KP * error + BRAKE_KI * integral + BRAKE_KD * derivative;
      controlSignals.brakeControl = constrain(controlSignals.brakeControl, 0, 255);

      if (abs(error) < 5) {
        controlSignals.brakeControl = 0;
      }

      xQueueOverwrite(controlQueue, &controlSignals);

      lastError = error;
    }
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

void actuatorTask(void *pvParameters) {
  SensorData sensorData;
  ControlSignals controlSignals;
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(20); // 50Hz

  xLastWakeTime = xTaskGetTickCount();

  for(;;) {
    if (xQueuePeek(sensorQueue, &sensorData, 0) == pdTRUE &&
        xQueuePeek(controlQueue, &controlSignals, 0) == pdTRUE) {
      
      // Speed control
      if (sensorData.pedalValue >= 375) {
        int mappedValue = map(sensorData.pedalValue, 0, 1023, 0, 255);
        ledcWrite(0, mappedValue);
      } else {
        ledcWrite(0, abs((int)controlSignals.speedControl));
      }

      // Brake control
      if (controlSignals.brakeControl > 0) {
        ledcWrite(1, (int)controlSignals.brakeControl);
        ledcWrite(2, 0);
      } else {
        ledcWrite(1, 0);
        ledcWrite(2, (int)controlSignals.brakeControl);
      }
    }
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

void serialTask(void *pvParameters) {
  SensorData sensorData;
  ControlSignals controlSignals;
  Setpoints setpoints;
  char incomingByte;
  String inputString = "";
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(100); // 10Hz

  xLastWakeTime = xTaskGetTickCount();

  for(;;) {
    // Read serial input
    while (Serial.available() > 0) {
      incomingByte = Serial.read();
      if (incomingByte == '\n') {
        int commaIndex = inputString.indexOf(',');
        if (commaIndex != -1) {
          setpoints.speedSetpoint = inputString.substring(0, commaIndex).toFloat();
          setpoints.brakeSetpoint = inputString.substring(commaIndex + 1).toFloat();
          xQueueOverwrite(setpointQueue, &setpoints);
        }
        inputString = "";
      } else {
        inputString += incomingByte;
      }
    }

    // Print debug information
    if (xQueuePeek(sensorQueue, &sensorData, 0) == pdTRUE &&
        xQueuePeek(controlQueue, &controlSignals, 0) == pdTRUE &&
        xQueuePeek(setpointQueue, &setpoints, 0) == pdTRUE) {
      
      Serial.print("Speed Setpoint: ");
      Serial.print(setpoints.speedSetpoint);
      Serial.print(" Brake Setpoint: ");
      Serial.print(setpoints.brakeSetpoint);
      Serial.print(" Current Pos: ");
      Serial.print(sensorData.potValue);
      Serial.print(" Speed Control: ");
      Serial.print(controlSignals.speedControl);
      Serial.print(" Brake Control: ");
      Serial.println(controlSignals.brakeControl);
    }

    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}