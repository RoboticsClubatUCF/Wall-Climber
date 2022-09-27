import serial


def main():
    ser = serial.Serial('/dev/rfcommm0', 38400) #open the serial port
    print(ser.name) #print name of used port
    ser.write('Hello World') #print string in port
    ser.close() #close serial port


main()
