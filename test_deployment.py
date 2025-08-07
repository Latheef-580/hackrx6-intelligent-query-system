#!/usr/bin/env python3
"""
Test script for deployed HackRx 6.0 system
Replace YOUR_APP_URL with your actual Render URL
"""

import requests
import json
import time

# Configuration - REPLACE WITH YOUR ACTUAL URL
DEPLOYED_URL = "https://hackrx6-intelligent-query-system.onrender.com"  # Replace this
TEAM_TOKEN = "b101776f72f459eca15614eb73a6f17efe85d475b21adb16c794068573018565"

def test_deployed_health():
    """Test health endpoint of deployed system"""
    print("🔍 Testing deployed health check...")
    try:
        response = requests.get(f"{DEPLOYED_URL}/health", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Deployed system is healthy!")
            print(f"Components: {data['components']}")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to deployed system: {e}")
        return False

def test_deployed_endpoint():
    """Test main endpoint of deployed system"""
    print("\n🚀 Testing deployed main endpoint...")
    
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TEAM_TOKEN}"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{DEPLOYED_URL}/hackrx/run",
            json=payload,
            headers=headers,
            timeout=120  # 2 minutes timeout for processing
        )
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Deployed system working!")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 Deployed HackRx 6.0 System Test")
    print("=" * 50)
    print(f"Testing URL: {DEPLOYED_URL}")
    print("=" * 50)
    
    # Test health
    health_ok = test_deployed_health()
    if not health_ok:
        print("\n❌ Deployed system is not responding")
        print("💡 Check your deployment on Render dashboard")
        return
    
    # Test main endpoint
    endpoint_ok = test_deployed_endpoint()
    
    # Summary
    print("\n" + "=" * 50)
    if health_ok and endpoint_ok:
        print("🎉 Deployed system is working perfectly!")
        print(f"✅ Your webhook URL is ready: {DEPLOYED_URL}/hackrx/run")
        print("📝 You can now submit this URL to the hackathon!")
    else:
        print("⚠️  Deployed system has issues")
        print("💡 Check the Render logs for errors")

if __name__ == "__main__":
    main() 