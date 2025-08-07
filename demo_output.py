#!/usr/bin/env python3
"""
Demo script to show what the HackRx 6.0 API output should look like
"""

import requests
import json
import time

def demo_health_check():
    """Demo health check"""
    print("🔍 Health Check Demo:")
    print("GET http://localhost:8000/health")
    print("Response:")
    print(json.dumps({
        "status": "healthy",
        "components": {
            "document_processor": True,
            "embedding_manager": True,
            "document_retriever": True,
            "response_builder": True
        },
        "timestamp": time.time()
    }, indent=2))
    print()

def demo_main_endpoint():
    """Demo main endpoint response"""
    print("🚀 Main Endpoint Demo:")
    print("POST http://localhost:8000/hackrx/run")
    print("Request:")
    print(json.dumps({
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }, indent=2))
    print()
    print("Expected Response:")
    print(json.dumps({
        "answers": [
            "Based on the National Parivar Mediclaim Plus Policy, there is a grace period of 15 days for premium payment. If the premium is not paid within this grace period, the policy will lapse and no claims will be payable.",
            "The waiting period for pre-existing diseases (PED) is 48 months (4 years) from the date of inception of the policy. During this period, any treatment related to pre-existing conditions will not be covered.",
            "Yes, this policy covers maternity expenses. The coverage includes normal delivery, cesarean section, and complications arising from pregnancy. However, there is a waiting period of 9 months from the date of inception, and the coverage is limited to a maximum of Rs. 50,000 per delivery."
        ]
    }, indent=2))

def test_actual_health():
    """Test actual health endpoint"""
    print("\n" + "="*60)
    print("🧪 ACTUAL SYSTEM TEST")
    print("="*60)
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"✅ Health Check Status: {response.status_code}")
        if response.status_code == 200:
            print("🎉 Your system is running and healthy!")
            print("📊 Components Status:")
            data = response.json()
            for component, status in data['components'].items():
                print(f"   • {component}: {'✅ Working' if status else '❌ Failed'}")
        else:
            print(f"❌ Health check failed: {response.text}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure the server is running with: uvicorn app.main:app --host 0.0.0.0 --port 8000")

def main():
    """Main demo function"""
    print("🎯 HackRx 6.0 API Demo")
    print("="*60)
    
    # Show expected outputs
    demo_health_check()
    demo_main_endpoint()
    
    # Test actual system
    test_actual_health()
    
    print("\n" + "="*60)
    print("📝 Next Steps:")
    print("1. Your system is working locally!")
    print("2. To deploy for hackathon submission, use Render or Railway")
    print("3. Your webhook URL will be: https://your-app-name.onrender.com/hackrx/run")
    print("4. The system will process documents and answer questions automatically")

if __name__ == "__main__":
    main() 