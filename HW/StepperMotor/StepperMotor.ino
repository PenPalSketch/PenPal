const byte STEP_PIN = 2;
const byte DIR_PIN = 5;
const byte ENABLE_PIN = 8;

void setup() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);

  digitalWrite(STEP_PIN, LOW);
  digitalWrite(ENABLE_PIN, LOW); // LOW enables the drivers
}

void moveMotor(bool direction, int steps) {
  digitalWrite(DIR_PIN, direction);
  delayMicroseconds(10); // Let direction settle before stepping

  for (int i = 0; i < steps; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(2000);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(2000);
  }
}

void loop() {
  moveMotor(HIGH, 200); // Move one way
  delay(1000);

  moveMotor(LOW, 200);  // Move back
  delay(1000);
}
