#!/usr/bin/env python3
"""
Simple test script to demonstrate the HackRx 6.0 API working
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
TEAM_TOKEN = "b101776f72f459eca15614eb73a6f17efe85d475b21adb16c794068573018565"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        print(f"📊 Components: {data['components']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_simple_request():
    """Test with a simple request"""
    print("\n🚀 Testing simple request...")
    
    # Simple test payload
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": ["What is the grace period for premium payment?"]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TEAM_TOKEN}"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        print(f"⏱️  Response time: {response.elapsed.total_seconds():.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Request successful!")
            print(f"📝 Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"📝 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 Simple HackRx 6.0 Test")
    print("=" * 40)
    
    # Test health
    health_ok = test_health()
    if not health_ok:
        print("❌ Server is not responding")
        return
    
    # Test simple request
    request_ok = test_simple_request()
    
    # Summary
    print("\n" + "=" * 40)
    if health_ok and request_ok:
        print("🎉 All tests passed! Your system is working!")
    else:
        print("⚠️  Some tests failed")

if __name__ == "__main__":
    main() 