# File Setup
ev3_slam installed on the lego ev3
pc_slam installed on the pc

# Running
- Before running, ensure that the ev3 and pc are paired with bluetooth but not connected
- Make sure to run 'sudo rfcomm bind 0 00:17:E9:XX:XX:XX 1' prior to running the code to set up communication channel (replace with MAC address of ev3)
- Note that the pc_slam needs sudo permission during run for bluetooth communication
