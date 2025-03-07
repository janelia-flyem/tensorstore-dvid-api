import json
import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data

def test_dataset_not_found():
    response = client.get("/api/node/dummy_uuid/nonexistent_dataset/info")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

def test_configure_dataset():
    # Basic TensorStore spec for testing
    test_spec = {
        "driver": "array",
        "dtype": "uint8",
        "array": [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        "chunk_layout": {
            "chunk_shape": [1, 1, 1]
        }
    }
    
    response = client.post(
        "/api/config/dataset?dataset_name=test_dataset", 
        json=test_spec
    )
    assert response.status_code == 200
    
    # Verify dataset was added
    response = client.get("/api/config/datasets")
    assert response.status_code == 200
    assert "test_dataset" in response.json()["datasets"]

# More comprehensive tests would require mocking the TensorStore API
# since we can't easily create real TensorStore objects in a test environment