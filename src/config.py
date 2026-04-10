
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MongoDBConfig:
    """MongoDB connection configuration"""
    host: str
    port: int
    username: str
    password: str
    database: str
    collection: str
    auth_source: str
    
    @property
    def connection_string(self) -> str:
        return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?authSource={self.auth_source}"
    
    @property
    def uri_safe(self) -> str:
        return f"mongodb://{self.host}:{self.port}/{self.database}"


@dataclass
class FlaskConfig:
    env: str
    host: str
    port: int
    debug: bool
    log_level: str


@dataclass
class AppConfig:
    mongodb: MongoDBConfig
    flask: FlaskConfig
    model_path: str
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        
        
        mongodb = MongoDBConfig(
            host=os.getenv('MONGO_HOST', 'mongo'),
            port=int(os.getenv('MONGO_PORT', 27017)),
            username=os.getenv('MONGO_USERNAME', 'root'),
            password=os.getenv('MONGO_PASSWORD', 'password'),
            database=os.getenv('MONGO_DATABASE', 'mle_db'),
            collection=os.getenv('MONGO_COLLECTION', 'predictions'),
            auth_source=os.getenv('MONGO_AUTH_SOURCE', 'admin')
        )
        
    
        flask = FlaskConfig(
            env=os.getenv('FLASK_ENV', 'production'),
            host=os.getenv('FLASK_HOST', '0.0.0.0'),
            port=int(os.getenv('FLASK_PORT', 55566)),
            debug=os.getenv('DEBUG_MODE', 'False').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO')
        )
        
     
        model_path = os.getenv('MODEL_PATH', 'experiments/log_reg.sav')
        
        return cls(
            mongodb=mongodb,
            flask=flask,
            model_path=model_path
        )



config = AppConfig.from_env()
