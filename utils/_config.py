# config.py


import os

def get_config():
    return {
        "model_path": os.getenv("MODEL_PATH", "models/your_model.pkl"),
        "catalog_path": os.getenv("CATALOG_PATH", "data/catalog.json"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "top_k": int(os.getenv("TOP_K", 5)),
        "debug": bool(int(os.getenv("DEBUG", 0)))
    }

class Config:
    """
    This class stores the configuration for your application. You can load different configurations
    based on the environment (development, production, etc.) and extend it as necessary.
    """
    # General settings
    APP_NAME = "MyApp"
    DEBUG = os.getenv("DEBUG", "False") == "True"  # This checks if the DEBUG environment variable is set to "True"
    
    # Database settings
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", 5432)  # Default to PostgreSQL
    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_NAME = os.getenv("DB_NAME", "myapp_db")

    # API keys and external service configurations
    API_KEY = os.getenv("API_KEY", "your-api-key-here")
    API_URL = os.getenv("API_URL", "https://api.example.com")

    # Other settings
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")
    
    @classmethod
    def get_config(cls):
        """
        This method returns the current configuration. You can expand this to load
        different settings based on environment (e.g., production or development).
        """
        return {
            "app_name": cls.APP_NAME,
            "debug": cls.DEBUG,
            "db_host": cls.DB_HOST,
            "db_port": cls.DB_PORT,
            "db_user": cls.DB_USER,
            "db_password": cls.DB_PASSWORD,
            "db_name": cls.DB_NAME,
            "api_key": cls.API_KEY,
            "api_url": cls.API_URL,
            "secret_key": cls.SECRET_KEY,
        }

# Optional: Add more specific configuration classes for development, production, etc.
class DevelopmentConfig(Config):
    DEBUG = True
    DB_HOST = "localhost"

class ProductionConfig(Config):
    DEBUG = False
    DB_HOST = "prod-db-server"
