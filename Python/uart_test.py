import serial

ser = serial.Serial("COM4", 921600, timeout=1)

print("Listening...")

while True:
    data = ser.readline()

    if data:
        print(repr(data))