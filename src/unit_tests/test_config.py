
import os
import pytest
from dotenv import load_dotenv


load_dotenv()


def test_config_loads_from_env():
    """Test that configuration loads from environment variables"""
    from config import config
    
    assert config is not None
    assert config.mongodb is not None
    assert config.flask is not None


def test_mongodb_config():
    """Test MongoDB configuration"""
    from config import config
    
    mongo_config = config.mongodb
    
   
    assert mongo_config.host
    assert mongo_config.port > 0
    assert mongo_config.username
    assert mongo_config.password
    assert mongo_config.database
    assert mongo_config.collection
    assert mongo_config.auth_source


def test_mongodb_connection_string():
    """Test that connection string is properly formatted"""
    from config import config
    
    mongo_config = config.mongodb
    conn_str = mongo_config.connection_string
    
  
    assert "mongodb://" in conn_str
    assert f"{mongo_config.username}:" in conn_str
    assert f"@{mongo_config.host}:{mongo_config.port}" in conn_str
    assert f"authSource={mongo_config.auth_source}" in conn_str
    
  
    safe_uri = mongo_config.uri_safe
    assert mongo_config.password not in safe_uri


def test_flask_config():
    """Test Flask configuration"""
    from config import config
    
    flask_config = config.flask
    
    assert flask_config.host
    assert flask_config.port > 0
    assert flask_config.env in ['development', 'production']
    assert flask_config.log_level


def test_model_path():
    """Test model path configuration"""
    from config import config
    
    assert config.model_path
    assert isinstance(config.model_path, str)
    assert len(config.model_path) > 0


def test_no_hardcoded_passwords():
    """Verify no hardcoded passwords in config module"""
    import config as config_module
    
  
    with open(config_module.__file__, 'r') as f:
        source = f.read()
    
    
    sensitive_patterns = [
        "mongodb://root:",
        "mongodb_password =",
        "MONGO_PASSWORD =",
    ]
    
    for pattern in sensitive_patterns:
        # Allow in comments/docstrings
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if pattern in line and not line.strip().startswith('#'):
                # This would indicate hardcoded credential
                assert False, f"Found hardcoded credential pattern: {pattern}"


def test_env_file_ignored():
    """Verify .env is in .gitignore"""
    import sys
    
    # Get project root (parent of src)
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    gitignore_path = os.path.join(project_root, '.gitignore')
    
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()
    
    # Should ignore .env files
    assert '.env' in gitignore_content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
