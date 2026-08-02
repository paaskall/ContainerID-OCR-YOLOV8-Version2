class DatabaseConfig:
    MYSQL_HOST = "localhost"
    MYSQL_PORT = 3306
    MYSQL_USER = "admin"
    MYSQL_PASSWORD = "passwordku123" 
    MYSQL_DATABASE = "autogate_system"
    
    POOL_SIZE = 5
    MAX_OVERFLOW = 10
    POOL_RECYCLE = 3600
    
    VEHICLES_TABLE = "vehicles"
    CONTAINERS_TABLE = "containers" 
    TRUCKS_TABLE = "trucks"
    ACCESS_LOGS_TABLE = "access_logs"
    DETECTION_LOGS_TABLE = "detection_logs"
    WEIGHT_LOGS_TABLE = "weight_logs"
    WEIGH_SESSIONS_TABLE = "weigh_sessions"
    WEIGH_SESSION_DETECTIONS_TABLE = "weigh_session_detections"
    CONTAINER_TRIPS_TABLE = "container_trips"

db_config = DatabaseConfig()