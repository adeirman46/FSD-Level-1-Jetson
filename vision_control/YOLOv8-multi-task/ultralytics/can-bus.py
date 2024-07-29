import can
from pynput import keyboard



     
# Define the message ID to filter
MESSAGE_ID_RPM = 0x47
MESSAGE_ID_SPEED  = 0x55

# Flag to control the loop
running = True

# Function to handle key press events
def on_press(key):
    global running
    try:
        if key.char == 'q':
            running = False
    except AttributeError:
        pass

# Function to handle key release events
def on_release(key):
    pass

# Set up the CAN interface
bus = can.interface.Bus(interface='socketcan', channel='can0', bitrate=500000)

# Set up the keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print(f"Listening for CAN messages with ID on can0. Press 'q' to quit.")

try:
    while running:
        msg = bus.recv(1.0)  # Timeout after 1 second
        if msg is not None and msg.arbitration_id == MESSAGE_ID_RPM:
                third_byte_rpm = msg.data[1]
                fifth_byte_rpm = msg.data[2]
                combined_data_rpm = (third_byte_rpm << 8) | fifth_byte_rpm  # Combine the bytes
                #speed = (0.001*combined_data)-9.25372+10
                rpm = (combined_data_rpm*3.6)+990
        elif msg is not None and msg.arbitration_id == MESSAGE_ID_SPEED:
                third_byte_speed = msg.data[3]
                fifth_byte_speed = msg.data[5]
                combined_data_speed = (third_byte_speed) # Combine the bytes
                #speed = (0.001*combined_data)-9.25372+10
                speed = (0.2829*combined_data_speed + 0.973)
                print(f'RPM : {rpm:0.0f}, SPEED = {speed:0.0f}')
        
    
            
except KeyboardInterrupt:
    pass
finally:
    listener.stop()

print("Program terminated.")