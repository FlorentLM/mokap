// Trigger firmware with full-range frequency and duty cycle support via millis()

const long SERIAL_BAUD_RATE = 115200;
const String FIRMWARE_ID = "PWM_TRIGGER_MILLIS_V1.0";
const int MAX_PINS = 20;

// Global state vars
bool isPinActive[MAX_PINS];
bool pinState[MAX_PINS];
unsigned long periodMillis[MAX_PINS];
unsigned long highTimeMillis[MAX_PINS];
unsigned long previousMillis[MAX_PINS];

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  // Initialize all pins to inactive state
  for (int i = 0; i < MAX_PINS; i++) {
    isPinActive[i] = false;
    pinState[i] = LOW;
  }
}

void loop() {
  //check for incoming commands from the host
  handleSerialCommands();
  // Update the state of all active pins (non-blocking)
  updatePinStates();
}

void handleSerialCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() > 0) {
      processCommand(command);
    }
  }
}

void updatePinStates() {
  unsigned long currentMillis = millis();

  for (int pin = 0; pin < MAX_PINS; pin++) {
    // only process pins that have been activated by a START command
    if (isPinActive[pin]) {
      // check if it's time to toggle pin state
      unsigned long interval = pinState[pin] ? highTimeMillis[pin] : (periodMillis[pin] - highTimeMillis[pin]);

      if (currentMillis - previousMillis[pin] >= interval) {
        previousMillis[pin] = currentMillis;
        pinState[pin] = !pinState[pin];
        digitalWrite(pin, pinState[pin]);
      }
    }
  }
}

void processCommand(String cmd) {

  if (cmd.equalsIgnoreCase("PING")) {
    Serial.println("PONG");
  }

  else if (cmd.equalsIgnoreCase("ID?")) {
    Serial.println(FIRMWARE_ID);
  }

  else if (cmd.startsWith("START")) {
    int firstSpace = cmd.indexOf(' ');
    int secondSpace = cmd.indexOf(' ', firstSpace + 1);
    int thirdSpace = cmd.indexOf(' ', secondSpace + 1);

    if (firstSpace != -1 && secondSpace != -1 && thirdSpace != -1) {
      int pin = cmd.substring(firstSpace + 1, secondSpace).toInt();
      float frequency = cmd.substring(secondSpace + 1, thirdSpace).toFloat();
      int dutyCycle = cmd.substring(thirdSpace + 1).toInt();

      if (pin >= 0 && pin < MAX_PINS && frequency > 0 && dutyCycle >= 0 && dutyCycle <= 100) {
        periodMillis[pin] = (unsigned long)(1000.0 / frequency);
        highTimeMillis[pin] = (unsigned long)(periodMillis[pin] * (dutyCycle / 100.0));
        previousMillis[pin] = millis();
        pinState[pin] = HIGH;
        isPinActive[pin] = true;

        pinMode(pin, OUTPUT);
        digitalWrite(pin, HIGH);

        Serial.println("OK");
      } else {
        Serial.println("ERROR: Invalid parameters for START command.");
      }
    } else {
      Serial.println("ERROR: Malformed START command. Expected: START <pin> <freq> <duty>");
    }
  }

  else if (cmd.startsWith("STOP")) {
    int spaceIndex = cmd.indexOf(' ');

    if (spaceIndex != -1) {
      int pin = cmd.substring(spaceIndex + 1).toInt();

      if (pin >= 0 && pin < MAX_PINS) {
        // deactivate the pin and set it to LOW state
        isPinActive[pin] = false;
        pinMode(pin, OUTPUT);
        digitalWrite(pin, LOW);
        Serial.println("OK");
      } else {
        Serial.println("ERROR: Invalid pin for STOP command.");
      }
    } else {
      Serial.println("ERROR: Malformed STOP command.");
    }
  }
  else {
    Serial.println("ERROR: Unknown command.");
  }
}