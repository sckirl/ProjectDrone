import serial
import time

class Wireless():
    def __init__(self, PORT, BAUD) -> None:
        self.PORT = PORT
        self.BAUD = BAUD

        self.ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)  # wait for connection
        print("Bluetooth Connection Successful")

    def sendMessage(self, message):
        self.ser.write((message + "\n").encode())

    def close(self):
        self.ser.close()


def testRun():
    # test run
    bl = Wireless("/dev/cu.ESP32ESP32", 9600)

    while 1:
        bl.sendMessage(input("> "))

    bl.close()

