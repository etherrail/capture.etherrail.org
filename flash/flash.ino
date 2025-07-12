#define FLASH_PIN 3

void setup() {
	pinMode(FLASH_PIN, OUTPUT);
}

void loop() {
	if (millis() % 20 < 2) {
		digitalWrite(FLASH_PIN, HIGH);
		delayMicroseconds(700);
		digitalWrite(FLASH_PIN, LOW);

		delay(5);
	}
}
