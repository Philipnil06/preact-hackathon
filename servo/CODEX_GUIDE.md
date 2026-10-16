# Guide för Codex — Axiometa Genesis Mini

Detta projekt kör på en **Axiometa Genesis Mini** (ESP32-S3). Här är reglerna du MÅSTE följa när du ändrar firmware.

## Pin-namn
Använd ALLTID board-alias, aldrig råa GPIO-siffror:
```cpp
// RÄTT
#define S1_PIN P1_IO0
#define S2_PIN P1_IO1

// FEL — kompilerar men ger fel beteende
#define S1_PIN 15
```
Tillgängliga alias per port: `P1_IO0`, `P1_IO1`, `P1_IO2` (och P2/P3/P4 om fler portar används).

## Servo-PWM
Servona drivs med `analogWrite()` — ingen extern servo-bibliotek behövs:
- Frekvens: 50 Hz
- Upplösning: 12-bit (duty 0–4095)
- Min puls: 500 µs → duty = 500*4095/20000 ≈ 102
- Max puls: 2500 µs → duty = 2500*4095/20000 ≈ 512

Försök INTE byta till `<ESP32Servo.h>` eller liknande — `analogWrite()` fungerar och är låst till detta projekt.

## Serial
- Använd alltid `Serial.begin(115200)`
- Lägg INTE till `while (!Serial)` — det hänger ESP32-S3
- Skriv aldrig `delay()` i `loop()` — använd `millis()` för timing

## Serieformat (browser → board)
Browsern skickar kommandon på formen:
```
S1:90\n
S2:45\n
S3:120\n
```
Boarden svarar med samma format. Ändra INTE detta protokoll utan att också uppdatera `browser/Servo Control.html`.

## Vad du FÅR ändra
- Logik i `loop()` och `setup()`
- Lägga till nya kommandon i Serial-parsern
- Lägga till nya funktioner

## Vad du INTE får ändra
- Pin-alias (använd P1_IO0 etc.)
- PWM-upplösning och frekvens för servona
- `Serial.begin(115200)` och frånvaron av `while (!Serial)`
- Serieprotokollets format (S1:/S2:/S3:)
