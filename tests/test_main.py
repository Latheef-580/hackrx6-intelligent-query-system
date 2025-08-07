import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json

from app.main import app

client = TestClient(app)

class TestMainEndpoints:
    """Test cases for main API endpoints"""
    
    def test_root_endpoint(self):
        """Test the root health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "active"
    
    def test_health_endpoint(self):
        """Test the detailed health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
    
    def test_hackrx_run_missing_auth(self):
        """Test hackrx/run endpoint without authentication"""
        payload = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is the coverage?"]
        }
        response = client.post("/hackrx/run", json=payload)
        assert response.status_code == 401
    
    def test_hackrx_run_invalid_payload(self):
        """Test hackrx/run endpoint with invalid payload"""
        headers = {"Authorization": "Bearer test_token"}
        
        # Test missing documents
        payload = {"questions": ["What is the coverage?"]}
        response = client.post("/hackrx/run", json=payload, headers=headers)
        assert response.status_code == 400
        
        # Test missing questions
        payload = {"documents": "https://example.com/test.pdf"}
        response = client.post("/hackrx/run", json=payload, headers=headers)
        assert response.status_code == 400
        
        # Test empty questions
        payload = {"documents": "https://example.com/test.pdf", "questions": []}
        response = client.post("/hackrx/run", json=payload, headers=headers)
        assert response.status_code == 400
    
    @patch('app.main.document_processor')
    @patch('app.main.document_retriever')
    @patch('app.main.response_builder')
    def test_hackrx_run_success(self, mock_response_builder, mock_retriever, mock_processor):
        """Test successful hackrx/run endpoint call"""
        # Mock the components
        mock_processor.process_document = AsyncMock()
        mock_processor.process_document.return_value = Mock()
        
        mock_retriever.build_vector_store = AsyncMock()
        mock_retriever.build_vector_store.return_value = "test_store_id"
        
        mock_retriever.retrieve_relevant_clauses = AsyncMock()
        mock_retriever.retrieve_relevant_clauses.return_value = []
        
        mock_response_builder.generate_response = AsyncMock()
        mock_response = Mock()
        mock_response.answer = "Test answer"
        mock_response_builder.generate_response.return_value = mock_response
        
        payload = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is the coverage?"]
        }
        headers = {"Authorization": "Bearer test_token"}
        
        response = client.post("/hackrx/run", json=payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "answers" in data
        assert len(data["answers"]) == 1
        assert data["answers"][0] == "Test answer"

class TestAuthentication:
    """Test authentication functionality"""
    
    def test_valid_token(self):
        """Test with valid token format"""
        headers = {"Authorization": "Bearer valid_token_12345"}
        response = client.get("/health", headers=headers)
        assert response.status_code == 200
    
    def test_invalid_token_format(self):
        """Test with invalid token format"""
        headers = {"Authorization": "invalid_token"}
        response = client.get("/health", headers=headers)
        assert response.status_code == 200  # Health endpoint doesn't require auth
    
    def test_missing_token(self):
        """Test with missing token"""
        payload = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is the coverage?"]
        }
        response = client.post("/hackrx/run", json=payload)
        assert response.status_code == 401

if __name__ == "__main__":
    pytest.main([__file__]) 