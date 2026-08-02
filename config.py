import os

class Config:
    APP_NAME = "AutoGate System"
    APP_VERSION = "1.0.0"
    
    CAMERAS = [
        {"id": 0, "name": "Camera 1 - Entrance", "source": 0, "active": True},
        {"id": 1, "name": "Camera 2 - Exit", "source": 1, "active": True},
        {"id": 2, "name": "Camera 3 - Weight Area", "source": 2, "active": True},
        {"id": 3, "name": "Camera 4 - Backup 1", "source": 3, "active": False},
        {"id": 4, "name": "Camera 5 - Backup 2", "source": 4, "active": False},
        {"id": 5, "name": "Camera 6 - Overview", "source": 5, "active": True}
    ]
    
    MODEL_PATH = os.path.join("models", "best.pt")
    DATABASE_PATH = os.path.join("data", "database.db")
    
    # Hardware
    SCALE_PORT = "COM3"
    SCALE_BAUDRATE = 9600
    GATE_PORT = "COM4"
    
    # GUI Settings
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900
    THEME = "modern"