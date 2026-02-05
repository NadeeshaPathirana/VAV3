qdrant_api_key = VDARHSVR4YQIVUwIgRl--8Fv5V_7QWEps-Hi-NGZZeR1bC5e0y8d_A

qdrant_api_key_usage = curl \
    -X GET 'https://83a4aa67-c595-4f58-a5e3-f5d6eb107808.us-east4-0.gcp.cloud.qdrant.io:6333' \
    --header 'api-key: VDARHSVR4YQIVUwIgRl--8Fv5V_7QWEps-Hi-NGZZeR1bC5e0y8d_A'

qdrant_cluster_url = https://83a4aa67-c595-4f58-a5e3-f5d6eb107808.us-east4-0.gcp.cloud.qdrant.io

1. Select and set the python interpreter (3.11)

For CPU:
2. Use virtual env - VAV1_cpu1
3. Install CPU requirements file in CPU virtual env ---> pip install -r requirements.txt

For GPU:
2. Use virtual env - VAV1_gpu1
3. Install GPU requirements file in GPU virtual environment ---> pip install -r requirements.txt
4. For GPU - Install pytorch with - pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 (this is working wiht CUDA 11x) or try pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

For Testing:
2. Use virtual env - VAV1_test_gpu
3. Install TEST requirements file in TEST virtual environment ---> pip install -r requirements.txt
4. For GPU - Install pytorch with - pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 (this is working wiht CUDA 11x) or try pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
5. Install addition libraries required for the current testing purposes

6. Start Ollama local server -> Ollama serve
7. While still online, run the code in test.txt for sentence-transformers/all-MiniLM-L6-v2. Sentence transformer will be cached.

communicate with local LLM model - curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\": \"mistral\", \"prompt\": \"Hello, how are you?\"}"


Speechify tts
------------------
Authorization: Bearer <API_KEY or ACCESS_TOKEN>
ACCESS_TOKEN: {
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTkyNDg2NDIsImlzcyI6InNwZWVjaGlmeS1hcGkiLCJzY29wZSI6ImF1ZGlvOmFsbCB2b2ljZXM6cmVhZCIsInN1YiI6IjR3aFJTT1cybEhOMTJTQjdxYkQ1OWhUelJ2ZjEifQ.hECdUmhFI5rF-3FsEsFK8gWeOti066tL1tAZ6sXlPSA",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "audio:all voices:read"
}


