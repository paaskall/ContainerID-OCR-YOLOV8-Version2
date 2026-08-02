import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import sys
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AutoGate System - G2")
    app.setApplicationVersion("1.0.0")
    
    try:
        from src.gui.main_window import MainWindow
        window = MainWindow()
        # AIzaSyDgr5oSJad23oCsIL4xKcarshDzicdqlfk
        # AIzaSyBQYysppLE-I7tixviyzhXweUcq8cmh0Gg
        # AIzaSyCg6lp_9WChprjtVpz75qPvxE_mALSubgE
        GEMINI_API_KEY = "AIzaSyDgr5oSJad23oCsIL4xKcarshDzicdqlfk"
        if hasattr(window, 'plate_recognizer'):
            print("Testing Gemini API configuration...")
            success = window.plate_recognizer.set_gemini_api_key(GEMINI_API_KEY)
            if success:
                print("Gemini API configured successfully")
                import numpy as np
                test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
                
                if hasattr(window, 'gemini_status'):
                    window.gemini_status.setText("Gemini API: CONNECTED")
            else:
                print("Gemini API configuration failed")
                if hasattr(window, 'gemini_status'):
                    window.gemini_status.setText("Gemini API: FAILED")
        
        window.show()
        
        print("=" * 50)
        print("AutoGate System Started Successfully!")
        print("6 CCTV Cameras Configured") 
        print("Container & Truck ID Recognition Ready")
        print("System Status: READY")
        print("=" * 50)
        print("Instructions:")
        print("1. Click 'Start All Cameras' to begin streaming")
        print("2. Detections will appear in colored bounding boxes")
        print("3. Green: container_id, Yellow: container_id_vertical")
        print("4. Blue: truck_id, Magenta: plate_number")
        print("=" * 50)
        
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please make sure all required files are in place.")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"Failed to start application: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()