#!/usr/bin/env python3
import sys
import os
from pynput import mouse
from termcolor import colored

aimbot_enabled = False

def on_click(x, y, button, pressed):
    global aimbot_enabled
    if button == mouse.Button.button9 and pressed:  # Mouse button 4
        aimbot_enabled = not aimbot_enabled
        if aimbot_enabled:
            print(colored('[+] AIMBOT ENABLED', 'green'))
        else:
            print(colored('[!] AIMBOT DISABLED', 'red'))
    elif pressed and button == mouse.Button.button8:  # Mouse button 5
        print(colored('[!] EXITING...', 'yellow'))
        os._exit(0)

def main():
    print(colored('''
╔═══════════════════════════════════════════════╗
║      ULTRA-FAST HUMAN-LIKE AIMBOT            ║
║       • Reaction: < 50ms                     ║
║       • Headshot Rate: 85%                   ║
║       • Style Learning: Active               ║
╚═══════════════════════════════════════════════╝
''', "cyan"))
    
    # Import and create aimbot
    from lib.aimbot import Aimbot
    
    # Start mouse listener
    listener = mouse.Listener(on_click=on_click)
    listener.start()
    
    # Create and run aimbot
    aimbot = Aimbot(box_constant=320)
    aimbot.start(lambda: aimbot_enabled)

if __name__ == "__main__":
    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Check dependencies
    try:
        from pynput import mouse
    except ImportError:
        print(colored("[!] Install pynput: pip install pynput", "red"))
        sys.exit(1)
    
    try:
        import cv2
    except ImportError:
        print(colored("[!] Install opencv: pip install opencv-python", "red"))
        sys.exit(1)
    
    # Run main
    main()
