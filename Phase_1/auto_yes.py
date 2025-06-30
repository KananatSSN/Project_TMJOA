# This is for automatically click yes when 3dslicer ask if you want to continue without GPU

import pyautogui
import keyboard
import time

button_image_path = r"C:\Users\acer\Desktop\Project_TMJOA\Resource\yes_button.png"  # Adjust the path as necessary
confidence_level = 0.8  # Adjust the confidence level as needed

while True:
    # Check if q key is pressed to break the loop
    if keyboard.is_pressed('q'):
        print("q key pressed. Exiting...")
        break
    
    try:
        # Locate the button on the screen with a specified confidence level
        location = pyautogui.locateOnScreen(button_image_path, confidence=confidence_level)
        if location:
            pyautogui.click(location)
            print("Button clicked.")
    except pyautogui.ImageNotFoundException:
        print("Button not found on the screen. Retrying...")
    
    time.sleep(15)  # Wait for half a second before trying again to avoid high CPU usage