import serial
serialcomm = serial.Serial('/dev/rfcomm0', 9600)
serialcomm.reset_input_buffer()

while 1:
    try:
        serialcomm.write(bytes(str(67),'utf_8'))
        inVar = serialcomm.read_until().decode()
        print("OUT: " + str(67) + " IN: " + str(inVar)) 
    except KeyboardInterrupt:
        break
serialcomm.close()