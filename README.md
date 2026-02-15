# Sarkaar Saathi

> **Your Voice-First Government Companion**  
> Eliminating Middlemen, Empowering Citizens

[![AWS AI for Bharat Hackathon](https://img.shields.io/badge/AWS-AI%20for%20Bharat%20Hackathon-FF9900?style=for-the-badge&logo=amazon-aws)](https://aws.amazon.com)
[![Built with Amazon Bedrock](https://img.shields.io/badge/Powered%20by-Amazon%20Bedrock-232F3E?style=for-the-badge&logo=amazon-aws)](https://aws.amazon.com/bedrock/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

---

## 🎯 The Problem

**120 million Maharashtra citizens** lack easy access to government scheme information. Rural citizens face:

- 💰 **Middleman Exploitation**: Pay ₹500-₹5000 per service
- 📚 **Literacy Barriers**: Cannot read complex government websites
- 🤷 **Information Gap**: 70% don't know about schemes they're eligible for
- 🌐 **Digital Divide**: Limited access to text-based portals

**Result**: Citizens pay middlemen thousands of rupees for services that should be free.

---

## 💡 Our Solution

**Sarkaar Saathi** is a voice-first conversational AI companion that guides Indian citizens through natural conversations to discover and access government schemes.

### 🎙️ Voice-First Design
Speak naturally in **Hindi, Marathi, or English** - no reading required.

### 🤖 Conversational Intelligence
Not a chatbot - a true conversational AI powered by **Amazon Bedrock** that:
- Asks intelligent follow-up questions
- Maintains full conversation context
- Guides you from "I need help" to "Here's exactly what to do"

### 📊 Cost Efficiency
**₹3 per conversation** vs **₹5000 to middlemen** = **99% savings**

---

## 🏗️ Architecture

### AWS-Native Serverless Stack

```
┌─────────────────────────────────────────┐
│  Client (Web/Mobile)                    │
│  🎤 Microphone → 🔊 Speaker             │
└─────────────────────────────────────────┘
                    ↕ WebSocket
┌─────────────────────────────────────────┐
│  API Gateway WebSocket                  │
│  Bidirectional Streaming                │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│  AWS Lambda (Python 3.11)               │
│  • Connection Handler                   │
│  • Conversation Controller ⭐           │
│  • Transcribe Streaming                 │
│  • Polly Streaming                      │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│  AI Services                            │
│  • Amazon Bedrock (Claude 3) ⭐         │
│  • Amazon Transcribe (Hindi/Marathi)    │
│  • Amazon Polly (Aditi Voice)           │
└─────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────┐
│  Data Layer                             │
│  • DynamoDB (Conversation State)        │
│  • DynamoDB (Vector Embeddings)         │
│  • S3 (Scheme Documents)                │
└─────────────────────────────────────────┘
```

**Region**: `ap-south-1` (Mumbai) - DPDP Act 2023 Compliant

---

## 🚀 Key Innovations

### 1. Conversational AI, Not a Chatbot

**Traditional Chatbot:**
```
User: "Farmer schemes?"
Bot: [Lists 10 schemes]
User: [Overwhelmed, leaves]
```

**Sarkaar Saathi:**
```
User: "मुझे खेती के लिए मदद चाहिए"
Saathi: "बहुत अच्छा! आप कहाँ रहते हैं?"
User: "पुणे में"
Saathi: "समझ गया। आपके पास कितनी जमीन है?"
[Conversation continues naturally...]
```

### 2. RAG Architecture - Dynamic Knowledge

- **No hardcoded responses** - retrieves relevant schemes dynamically
- **Automatic adaptation** - add new schemes without code changes
- **Context-aware** - understands conversation history and citizen profile

### 3. Amazon Bedrock Intelligence

```python
# Claude 3 powers conversational reasoning
response = bedrock_runtime.invoke_model(
    modelId='anthropic.claude-3-haiku-20240307-v1:0',
    body=json.dumps({
        'max_tokens': 512,
        'temperature': 0.7,
        'messages': [{
            'role': 'user',
            'content': build_conversational_prompt(
                history=conversation['history'],
                schemes=relevant_schemes,
                phase=current_phase
            )
        }]
    })
)
```

### 4. Conversation Phase Management

Intelligent flow control through phases:
1. **Greeting** → Welcome and language selection
2. **Discovery** → Understand citizen's needs
3. **Eligibility** → Progressive criteria collection
4. **Documents** → Required document guidance
5. **Guidance** → Application steps + fraud warnings

---

## 📊 Cost Analysis

### MVP (1,000 conversations/month)

| Service | Monthly Cost |
|---------|--------------|
| Amazon Bedrock (Claude 3 Haiku) | $13.75 |
| Amazon Transcribe Streaming | $12.00 |
| Amazon Polly Neural | $4.00 |
| AWS Lambda + DynamoDB + S3 | $10.00 |
| Amazon API Gateway WebSocket | $3.50 |
| **Total** | **$43.25** |

**Cost per conversation: ₹3.60**

### Scale (100,000 conversations/month)

| Service | Monthly Cost |
|---------|--------------|
| Amazon Bedrock | $1,375 |
| Amazon Transcribe | $1,200 |
| Amazon Polly | $400 |
| Infrastructure | $685 |
| **Total** | **$3,660** |

**Cost per conversation: ₹3.10** (economies of scale!)

**ROI**: 99%+ savings vs ₹500-₹5000 paid to middlemen

---

## 🎬 Demo

### Live Conversation Example (Hindi)

```
🎤 User: "मुझे खेती के लिए कर्ज माफी योजना चाहिए"
   (I need a loan waiver scheme for farming)

🤖 Saathi: "बहुत अच्छा! मैं आपकी मदद कर सकता हूं। 
           क्या आप मुझे बता सकते हैं कि आप कहाँ रहते हैं?"
   (Great! I can help you. Can you tell me where you live?)

🎤 User: "मैं पुणे में रहता हूं"
   (I live in Pune)

🤖 Saathi: "समझ गया। और आपके पास कितनी जमीन है?"
   (I understand. And how much land do you have?)

🎤 User: "मेरे पास 3 एकड़ जमीन है"
   (I have 3 acres of land)

🤖 Saathi: "अच्छा। और आपका कर्ज कितना है?"
   (Good. And how much is your loan?)

🎤 User: "₹1 लाख"
   (₹1 lakh)

🤖 Saathi: "बहुत अच्छा! आप महात्मा ज्योतिबा फुले शेतकरी 
           कर्जमुक्ती योजना के लिए पात्र हैं। यह योजना 
           ₹1.5 लाख तक के कर्ज माफ करती है।"
   (Excellent! You're eligible for the Mahatma Jyotiba Phule 
    Shetkari Karjmukti Yojana. This scheme waives loans up to ₹1.5 lakh.)
```

[▶️ Watch Full Demo Video](To be updated) | [🎮 Try Live Demo](To be updated)

---

## 🛠️ Tech Stack

### AWS Services

- **Amazon Bedrock** - Claude 3 Sonnet/Haiku for conversational intelligence
- **Amazon Transcribe** - Real-time speech-to-text (Hindi/Marathi/English)
- **Amazon Polly** - Neural text-to-speech (Aditi voice for Hindi)
- **AWS Lambda** - Serverless compute (Python 3.11)
- **Amazon API Gateway** - WebSocket API for streaming
- **Amazon DynamoDB** - Conversation state + vector embeddings
- **Amazon S3** - Scheme document storage
- **Amazon CloudWatch** - Monitoring and logging
- **AWS IAM** - Security and access control

### Languages & Frameworks

- **Python 3.11** - Lambda functions
- **JavaScript** - Web client
- **HTML/CSS** - Demo interface

---

## 📁 Project Structure

```
sarkaar-saathi/
├── .kiro/specs/sarkaar-saathi/
│   ├── requirements.md          # 20 comprehensive requirements
│   ├── design.md                # Technical architecture & design
│   ├── tasks.md                 # Implementation roadmap
│   ├── blog.md                  # Technical blog post
│   ├── pitch_deck_outline.md    # Presentation slides
│   └── video_script.md          # Demo video script
├── lambda/
│   ├── connection_handler/      # WebSocket connection management
│   ├── conversation_controller/ # Core orchestrator (RAG + Bedrock)
│   ├── transcribe_streaming/    # Speech-to-text handler
│   ├── polly_streaming/         # Text-to-speech handler
│   └── embedding_generator/     # Scheme embedding generation
├── web/
│   ├── index.html               # Demo web interface
│   ├── app.js                   # WebSocket client
│   └── styles.css               # UI styling
├── schemes/
│   └── maharashtra/             # 50+ Maharashtra scheme documents
├── infrastructure/
│   ├── cloudformation/          # AWS infrastructure as code
│   └── terraform/               # Alternative IaC
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # End-to-end tests
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Prerequisites

- AWS Account with access to:
  - Amazon Bedrock (Claude 3 model access)
  - Amazon Transcribe, Polly, Lambda, API Gateway, DynamoDB, S3
- AWS CLI configured with credentials
- Python 3.11+
- Node.js 18+ (for web client)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/sarkaar-saathi.git
cd sarkaar-saathi
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure AWS credentials**
```bash
aws configure
# Set region to ap-south-1 (Mumbai)
```

4. **Deploy infrastructure**
```bash
cd infrastructure/cloudformation
./deploy.sh
```

5. **Upload scheme documents**
```bash
aws s3 sync schemes/ s3://sarkaar-saathi-schemes-bucket/
```

6. **Run web demo locally**
```bash
cd web
python -m http.server 8000
# Open http://localhost:8000
```

---

## 📖 Documentation

- **[Requirements Document](requirements.md)** - Detailed functional requirements
- **[Design Document](design.md)** - Architecture and technical design

---

## 🎯 Features

### ✅ MVP (Current)

- [x] Voice input/output in Hindi, Marathi, English
- [x] Conversational flow with state management
- [x] RAG-based scheme retrieval
- [x] Progressive eligibility discovery
- [x] Document guidance
- [x] Fraud warnings
- [x] 50+ Maharashtra schemes
- [x] WebSocket streaming
- [x] Web-based demo

---

## 🌍 Social Impact

### Immediate Impact (Maharashtra MVP)
- **120M** potential users
- **50+** schemes covered
- **3** languages supported
- **₹500-₹5000** saved per service

### Projected Impact (Year 1)
- **1M** conversations
- **₹50 crore** saved from middleman fees
- **500K** citizens connected to schemes
- **80%** conversation completion rate

### National Scale Potential
- **1.4B** Indians
- **28** states + 8 UTs
- **22** official languages
- **Thousands** of schemes

---

## 🔒 Security & Compliance

### DPDP Act 2023 Compliance

- ✅ Data residency in Mumbai region (ap-south-1)
- ✅ Encryption at rest (AWS KMS)
- ✅ Encryption in transit (TLS 1.3)
- ✅ IAM least privilege access
- ✅ 90-day audit logs (CloudWatch)
- ✅ Right to erasure (30-day data deletion)
- ✅ Data minimization principles

### Security Features

- API Gateway authentication
- Lambda execution roles with minimal permissions
- VPC endpoints for private connectivity
- Secrets Manager for credentials
- CloudWatch monitoring and alarms

---

## 📊 Performance Metrics

- **Streaming Latency**: < 500ms (transcription), < 2s (response)
- **Conversation Completion**: > 80%
- **Transcription Accuracy**: > 85% (Hindi/Marathi)
- **System Availability**: > 99.5%
- **Average Turns to Resolution**: < 10 turns

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

Built for the **AWS AI for Bharat Hackathon 2024**

- **[Aditya Belhekar]** - [GitHub](https://github.com/AdityaBelhekar)
- **[Pranav Panchal]** - [GitHub](https://github.com/pranavpanchal1326)

---

## 🙏 Acknowledgments

- **AWS** for providing Bedrock, Transcribe, Polly, and other services
- **Maharashtra Government** for scheme information
- **Rural citizens** who inspired this solution
- **AWS AI for Bharat Hackathon** organizers

---

## 📞 Contact

- **Email**: [belhekaraditya96@gmail.com]/[pranavpanchal1326@gmail.com]

---

<div align="center">

**Built with ❤️ for India**

**Let's eliminate middlemen and empower every Indian to access their rights.**

[⭐ Star this repo](https://github.com/AdityaBelhekar/Sarkaar-Saathi) | [🐛 Report Bug](https://github.com/AdityaBelhekar/Sarkaar-Saathi/issues) | [💡 Request Feature](https://github.com/AdityaBelhekar/Sarkaar-Saathi/issues)

</div>
