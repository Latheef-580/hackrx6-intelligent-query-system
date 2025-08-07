#!/usr/bin/env python3
"""
Test script for HackRx 6.0 Intelligent Query-Retrieval System
This script tests the system with the sample payload provided in the hackathon.
"""

import asyncio
import aiohttp
import json
import time
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEAM_TOKEN = "b101776f72f459eca15614eb73a6f17efe85d475b21adb16c794068573018565"

async def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Health check passed: {data['status']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status}")
                return False

async def test_hackrx_run():
    """Test the main hackrx/run endpoint"""
    print("\n🚀 Testing HackRx /run endpoint...")
    
    # Load sample payload
    payload_path = Path("sample_payload.json")
    if not payload_path.exists():
        print("❌ sample_payload.json not found!")
        return False
    
    with open(payload_path, 'r') as f:
        payload = json.load(f)
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {TEAM_TOKEN}"
    }
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/hackrx/run",
            json=payload,
            headers=headers
        ) as response:
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            
            if response.status == 200:
                data = await response.json()
                print(f"✅ Request successful!")
                print(f"📊 Received {len(data.get('answers', []))} answers")
                
                # Display first few answers
                for i, answer in enumerate(data.get('answers', [])[:3]):
                    print(f"   Q{i+1}: {answer[:100]}...")
                
                if len(data.get('answers', [])) > 3:
                    print(f"   ... and {len(data.get('answers', [])) - 3} more answers")
                
                return True
            else:
                error_text = await response.text()
                print(f"❌ Request failed: {response.status}")
                print(f"Error: {error_text}")
                return False

async def test_single_question():
    """Test the single question endpoint"""
    print("\n🧪 Testing single question endpoint...")
    
    headers = {
        "Authorization": f"Bearer {TEAM_TOKEN}"
    }
    
    params = {
        "document_url": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "question": "What is the grace period for premium payment?"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/test/single-question",
            params=params,
            headers=headers
        ) as response:
            
            if response.status == 200:
                data = await response.json()
                print(f"✅ Single question test passed!")
                print(f"Answer: {data.get('answer', 'No answer')[:100]}...")
                return True
            else:
                error_text = await response.text()
                print(f"❌ Single question test failed: {response.status}")
                print(f"Error: {error_text}")
                return False

async def main():
    """Main test function"""
    print("🧪 HackRx 6.0 System Test")
    print("=" * 50)
    
    # Check if server is running
    try:
        health_ok = await test_health_check()
        if not health_ok:
            print("\n❌ Server is not responding. Please start the server first:")
            print("   uvicorn app.main:app --host 0.0.0.0 --port 8000")
            return False
    except Exception as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print("Please start the server first:")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False
    
    # Run tests
    tests = [
        test_hackrx_run,
        test_single_question
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! System is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 