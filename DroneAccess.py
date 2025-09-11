import serial
import time
import struct

class Drone():
    def __init__(self, COM_PORT, BAUD_RATE=115200):
        self.COM_PORT = COM_PORT
        self.BAUD_RATE = BAUD_RATE  # This must match your INAV MSP port speed

        # --- MSP Protocol constants ---
        self.MSP_RAW_IMU = 102 # Command to request Gyro, Acc, and Mag data
        self.MSP_RAW_GPS = 106

    def construct_msp_request(self, command):
        """Constructs a simple MSPv1 request packet (with no payload)."""
        # Packet structure: $M< [size] [command] [checksum]
        header = b'$M<'
        payload_size = 0
        
        size_byte = struct.pack('<B', payload_size)
        command_byte = struct.pack('<B', command)
        
        # Checksum is XOR of size and command
        checksum = payload_size ^ command
        checksum_byte = struct.pack('<B', checksum)
        
        return header + size_byte + command_byte + checksum_byte

    def parse_imu_response(self, data):
        """Parses the 18-byte payload from an MSP_RAW_IMU response."""
        # The payload is 9 x 2-byte signed integers (int16_t)
        # Order: accX, accY, accZ, gyroX, gyroY, gyroZ, magX, magY, magZ
        # We only care about the gyro values for this script.
        try:
            # '<' for little-endian, 'h' for signed short (2 bytes)
            sensors = struct.unpack('<9h', data)
            acc_x, acc_y, acc_z = sensors[0:3]
            gyro_x, gyro_y, gyro_z = sensors[3:6]
            # mag_x, mag_y, mag_z = sensors[6:9] # We can ignore mag for now
            
            return gyro_x, gyro_y, gyro_z
        except struct.error:
            # This can happen if we get an incomplete packet
            return None, None, None
        
    def parse_gps_response(self, data):
        """Parses the 16-byte payload from an MSP_RAW_GPS response."""
        try:
            # Format: fix(byte), sats(byte), lat(int), lon(int), alt(short), speed(ushort), course(ushort)
            fix_type, num_sats, lat, lon, alt, speed, course = struct.unpack('<bbiiHHH', data)
            
            # Convert lat/lon to standard decimal degrees
            lat_dec = lat / 10000000.0
            lon_dec = lon / 10000000.0

            return fix_type, num_sats, lat_dec, lon_dec, alt
        except struct.error:
            return None, None, None, None, None

    def readGyro(self):
        print(f"--- INAV Gyroscope Reader ---")
        
        try:
            # Open the serial port connection
            with serial.Serial(self.COM_PORT, self.BAUD_RATE, timeout=1) as ser:
                print(f"Successfully connected to {self.COM_PORT}")
                print("Requesting gyroscope data... Press Ctrl+C to stop.")
                
                # 1. Send the request packet
                request_packet = self.construct_msp_request(self.MSP_RAW_IMU)
                ser.write(request_packet)
                
                # 2. Wait a very short moment for the response
                time.sleep(0.05)
                
                # 3. Read the response from the serial buffer
                response_data = ser.read_all()
                
                # A valid MSP_RAW_IMU response starts with '$M>' and has an 18-byte payload
                if response_data.startswith(b'$M>') and len(response_data) >= 23: # Header(3) + Size(1) + Cmd(1) + Payload(18)
                    payload = response_data[5:23] # Extract just the payload bytes
                    gyro_x, gyro_y, gyro_z = self.parse_imu_response(payload)
                    
                    if gyro_x is not None:
                        # returns the value in this order: [x, y, z]
                        
                        print(f"\rGyro X: {gyro_x: 6d} | Gyro Y: {gyro_y: 6d} | Gyro Z: {gyro_z: 6d}  ", end="")

                        return [gyro_x, gyro_y, gyro_z]

        except serial.SerialException as e:
            print(f"\nERROR: Could not connect to {self.COM_PORT}.")
            print(f"Details: {e}")
            print("Please check the following:")
            print("1. Is the drone plugged in via USB?")
            print("2. Is the COM port name correct?")
            print("3. Is the INAV-Configurator (or any other serial program) closed?")
        except KeyboardInterrupt:
            print("\n\nStopping data stream. Exiting.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

    def readGPS(self):
        """Connects to the drone and continuously reads GPS coordinate data."""
        print(f"--- INAV GPS Reader ---")
        try:
            with serial.Serial(self.COM_PORT, self.BAUD_RATE, timeout=1) as ser:
                print(f"Successfully connected to {self.COM_PORT}")
                print("Requesting GPS data... Press Ctrl+C to stop.")

                request_packet = self.construct_msp_request(self.MSP_RAW_GPS)
                ser.write(request_packet)
                time.sleep(0.1) # GPS updates are slower, so we can poll less frequently
                response_data = ser.read_all()
                
                # A valid MSP_RAW_GPS response starts with '$M>' and has a 16-byte payload
                if response_data.startswith(b'$M>') and len(response_data) >= 21: # Header(3) + Size(1) + Cmd(1) + Payload(16)
                    payload = response_data[5:21]
                    fix, sats, lat, lon, alt = self.parse_gps_response(payload)
                    
                    if fix is not None:

                        print(f"\rSats: {sats:2d} | Fix: {fix} | Lat: {lat:9.5f} | Lon: {lon:9.5f} | Alt: {alt}m  ", end="")

                        return [sats, fix, lat, lon, alt]

        except serial.SerialException as e:
            print(f"\nERROR: Could not connect to {self.COM_PORT}.\nDetails: {e}")
        except KeyboardInterrupt:
            print("\n\nStopping data stream. Exiting.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    test = Drone("/dev/tty.usbmodem0x80000001")
    test.readGPS()