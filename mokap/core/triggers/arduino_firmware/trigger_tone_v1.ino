// Trigger firmware with high-precision timing via tone()
// Supports frequencies >= 31 Hz and 50% duty cycle only

const long SERIAL_BAUD_RATE = 115200;
const String FIRMWARE_ID = "PWM_TRIGGER_TONE_V1.0";

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() > 0) {
      processCommand(command);
    }
  }
}

void processCommand(String cmd) {
   if (cmd.equalsIgnoreCase("PING")) {
    Serial.println("PONG");
    return;
  }

  else if (cmd.equalsIgnoreCase("ID?")) {
    Serial.println(FIRMWARE_ID);
  }

  else if (cmd.startsWith("START")) {
    int firstSpace = cmd.indexOf(' ');
    int secondSpace = cmd.indexOf(' ', firstSpace + 1);

    if (firstSpace != -1 && secondSpace != -1) {
      String pinStr = cmd.substring(firstSpace + 1, secondSpace);
      String freqStr = cmd.substring(secondSpace + 1);

      int pin = pinStr.toInt();
      // On AVR boards (Uno, Nano), tone() requires an *unsigned int* for frequency
      unsigned int frequency = freqStr.toInt();

      // the minimum frequency for tone() on Uno/Nano is 31 Hz
      if (pin > 0 && frequency >= 31) {
        tone(pin, frequency);
        Serial.println("OK");
      } else {
        Serial.println("ERROR: Invalid pin or frequency (must be >= 31 Hz).");
      }
    } else {
      Serial.println("ERROR: Malformed START command.");
    }
  }

  else if (cmd.startsWith("STOP")) {
    int spaceIndex = cmd.indexOf(' ');
    if (spaceIndex != -1) {
      String pinStr = cmd.substring(spaceIndex + 1);
      int pin = pinStr.toInt();

      noTone(pin);
      pinMode(pin, OUTPUT);
      digitalWrite(pin, LOW);
      Serial.println("OK");
    } else {
      Serial.println("ERROR: Malformed STOP command.");
    }
  }

  else {
    Serial.println("ERROR: Unknown command.");
  }
}