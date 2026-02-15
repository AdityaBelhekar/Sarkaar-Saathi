# Requirements Document: Sarkaar Saathi

## Executive Summary

Sarkaar Saathi is a voice-first, multilingual Digital Government Companion designed to help Indian citizens discover and access government schemes using AWS-native Generative AI services. The system addresses the critical problem of information asymmetry where citizens, particularly in rural areas, pay middlemen ₹500-₹5000 per service due to lack of awareness and digital literacy barriers.

### Problem Statement

- 120 million Maharashtra citizens lack easy access to government scheme information
- Rural citizens face literacy barriers (voice-first AI solves this)
- Middleman exploitation costs citizens ₹500-₹5000 per service
- Complex eligibility criteria and documentation requirements create confusion
- Limited digital accessibility in rural areas

### Solution Overview

A serverless, AWS-native conversational voice companion powered by Amazon Bedrock that uses Retrieval-Augmented Generation (RAG) to provide intelligent, personalized guidance on government schemes. The system is voice-first with continuous conversational interaction - NOT a text-based chatbot. It maintains conversation state across multiple turns, asks follow-up questions, and guides citizens through discovery, eligibility assessment, and application preparation. The system supports Hindi, Marathi, and English, with streaming voice input/output optimized for low-literacy users.

### Key Innovation: RAG vs Rule-Based Chatbot

**Why NOT Rule-Based Chatbot:**
- Rule-based systems require manual mapping of every question-answer pair
- Cannot handle natural language variations or context
- Brittle when schemes change or new schemes are added
- Cannot provide personalized, context-aware responses
- Limited to predefined conversation flows

**How RAG Improves Accuracy:**
- Retrieves relevant scheme information from knowledge base dynamically
- Bedrock models understand context and generate natural responses
- Automatically adapts to scheme updates without code changes
- Handles follow-up questions and multi-turn conversations
- Provides personalized explanations based on user context

**How Bedrock Enables Intelligent Guidance:**
- Proactive follow-up questions to determine eligibility
- Natural language understanding of citizen queries
- Context-aware document checklist generation
- Fraud warning detection and prevention
- Ranked scheme recommendations based on relevance

### AWS Technology Stack

**Core Services:**
- Amazon Bedrock (Claude 3 Sonnet/Haiku or Titan models) - Conversational reasoning engine
- Amazon Polly (Hindi Aditi voice) - Streaming text-to-speech
- Amazon Transcribe (Hindi language support) - Streaming speech-to-text
- AWS Lambda - Serverless compute
- Amazon API Gateway - WebSocket API for streaming conversations
- Amazon DynamoDB - Conversation state and scheme metadata storage
- Amazon S3 - Scheme document storage and embeddings
- Amazon CloudWatch - Monitoring and logging
- AWS IAM - Security and access control

**Region:** ap-south-1 (Mumbai) for data residency compliance

### MVP Scope (Maharashtra Focus)

**Phase 1 Features:**
- Streaming voice input/output in Hindi and Marathi via WebSocket
- Continuous conversational interaction with state management
- Intelligent conversation flow control (discovery → eligibility → documents → guidance)
- 50+ Maharashtra government schemes (agriculture, education, health, housing)
- Proactive follow-up questions and eligibility detection
- Document readiness checklist generation
- Basic fraud warnings
- Multi-turn conversation context maintenance

**Out of Scope for MVP:**
- Application submission
- Payment processing
- Document upload/verification
- Multi-state scheme coverage
- Mobile app (web-based MVP)

### Cost Estimates

**MVP (1,000 users/month):**
- Bedrock (Claude 3 Haiku): ~$15/month (100K input tokens, 20K output tokens per user)
- Transcribe: ~$12/month (1,000 minutes)
- Polly: ~$4/month (1M characters)
- Lambda: ~$5/month (within free tier mostly)
- DynamoDB: ~$2/month (on-demand)
- S3: ~$1/month
- API Gateway: ~$3.50/month
- **Total: ~$42.50/month (~₹3,500)**
- **Cost per user: ~$0.04 (₹3.50)**

**Scale (100,000 users/month):**
- Bedrock: ~$1,500/month
- Transcribe: ~$1,200/month
- Polly: ~$400/month
- Lambda: ~$150/month
- DynamoDB: ~$50/month
- S3: ~$10/month
- API Gateway: ~$350/month
- **Total: ~$3,660/month (~₹3,00,000)**
- **Cost per user: ~$0.037 (₹3.10)**

### Compliance Requirements

**DPDP Act 2023 (Digital Personal Data Protection Act):**
- Data storage in Mumbai region (ap-south-1)
- Encryption at rest and in transit
- User consent management
- Data minimization principles
- Right to erasure support

**Security:**
- IAM least privilege access
- VPC endpoints for private connectivity
- CloudWatch audit logging
- Secrets Manager for credentials
- API Gateway throttling and authentication

## Glossary

- **Sarkaar_Saathi**: The voice-first conversational digital government companion system
- **Citizen**: End user engaging in voice conversations about government schemes
- **Conversation_State**: Complete context of an ongoing conversation including history, extracted entities, and current phase
- **Conversation_Phase**: Current stage of conversation (greeting → discovery → eligibility → documents → guidance → closure)
- **Scheme**: Government welfare program or service
- **RAG_Engine**: Retrieval-Augmented Generation system combining vector search with LLM generation for contextual understanding
- **Bedrock_Service**: Amazon Bedrock managed service for foundation models providing conversational intelligence
- **Voice_Interface**: Combined streaming Transcribe (input) and Polly (output) for continuous voice interaction
- **Conversation_Controller**: Component that manages conversation flow and determines next actions
- **Eligibility_Detector**: Component that progressively discovers citizen eligibility through natural conversation
- **Document_Assistant**: Component that guides citizens through document requirements conversationally
- **Session_Manager**: Component that maintains conversation state across WebSocket connections
- **Scheme_Repository**: S3 and DynamoDB storage for scheme information
- **Embedding_Store**: Vector embeddings of scheme documents for semantic search
- **Query_Processor**: Lambda function that orchestrates RAG pipeline for each conversation turn
- **Response_Generator**: Bedrock-powered component that creates contextual conversational responses
- **Fraud_Detector**: Component that naturally weaves fraud warnings into conversations
- **API_Gateway**: WebSocket API endpoint for streaming voice conversations
- **CloudWatch_Monitor**: Logging and monitoring service for conversation observability

## Requirements

### Requirement 1: Streaming Voice Input Processing

**User Story:** As a citizen with limited literacy, I want to speak continuously and naturally, so that I can have a conversation without repeatedly pressing buttons.

#### Acceptance Criteria

1. WHEN a citizen connects via WebSocket, THE System SHALL establish a streaming audio connection
2. WHEN a citizen speaks in Hindi, THE Transcribe_Service SHALL stream partial transcripts in real-time
3. WHEN a citizen speaks in Marathi, THE Transcribe_Service SHALL stream partial transcripts in real-time
4. WHEN a citizen pauses speaking, THE System SHALL detect end-of-utterance and process the complete statement
5. WHEN audio quality is poor, THE System SHALL request the citizen to repeat without breaking the conversation flow

### Requirement 2: Conversational Flow Control

**User Story:** As a citizen, I want the system to guide me through a natural conversation, so that I don't need to know what questions to ask.

#### Acceptance Criteria

1. WHEN a citizen starts a conversation, THE System SHALL greet them and ask how it can help
2. WHEN a citizen provides information, THE System SHALL determine if more information is needed and ask one follow-up question at a time
3. WHEN enough information is collected, THE System SHALL transition naturally to the next conversation phase (discovery → eligibility → documents → guidance)
4. WHEN a conversation topic is complete, THE System SHALL ask if the citizen needs help with anything else
5. WHEN a citizen goes off-topic, THE System SHALL gently redirect the conversation while maintaining context

### Requirement 3: Conversation State Management

**User Story:** As a citizen, I want the system to remember what we've discussed, so that I don't have to repeat myself.

#### Acceptance Criteria

1. WHEN a citizen provides information, THE System SHALL store it in the conversation state in DynamoDB
2. WHEN generating responses, THE Bedrock_Service SHALL have access to full conversation history
3. WHEN a citizen refers to previously mentioned information, THE System SHALL understand the reference
4. WHEN a conversation is interrupted, THE System SHALL allow resumption from the same point within 30 minutes
5. WHEN a conversation completes, THE System SHALL save a summary for analytics while clearing sensitive data

### Requirement 4: RAG-Based Contextual Understanding

**User Story:** As a system architect, I want to use RAG for contextual understanding, so that the system provides accurate, conversation-aware responses that adapt to scheme changes.

#### Acceptance Criteria

1. WHEN processing citizen input, THE RAG_Engine SHALL generate embeddings and retrieve top 5 relevant scheme documents from Embedding_Store
2. WHEN scheme documents are retrieved, THE System SHALL pass them along with conversation history to Bedrock_Service
3. WHEN Bedrock_Service generates a response, THE System SHALL ensure it's contextually appropriate for the current conversation phase
4. WHEN scheme information is updated in S3, THE System SHALL regenerate embeddings within 1 hour
5. WHEN no relevant schemes are found, THE System SHALL ask clarifying questions rather than ending the conversation

### Requirement 5: Bedrock-Powered Conversational Intelligence

**User Story:** As a citizen, I want the system to understand my needs and guide me intelligently, so that I can get help without knowing what to ask.

#### Acceptance Criteria

1. WHEN generating responses, THE Bedrock_Service SHALL use Claude 3 Sonnet or Haiku model with conversation history
2. WHEN a citizen's intent is unclear, THE System SHALL ask one clarifying question at a time
3. WHEN explaining schemes, THE System SHALL provide information progressively based on conversation phase
4. WHEN detecting eligibility gaps, THE System SHALL proactively ask about missing criteria
5. WHEN a citizen shows confusion, THE System SHALL simplify explanations and provide examples

### Requirement 6: Progressive Eligibility Discovery

**User Story:** As a citizen, I want the system to naturally discover my eligibility through conversation, so that I don't feel interrogated.

#### Acceptance Criteria

1. WHEN a citizen expresses interest in a scheme, THE System SHALL ask one eligibility question at a time in natural conversation
2. WHEN eligibility information is provided, THE System SHALL acknowledge it and ask the next relevant question
3. WHEN enough criteria are collected, THE System SHALL determine eligibility and explain the result
4. WHEN a citizen is eligible for multiple schemes, THE System SHALL discuss them one at a time
5. WHEN a citizen is not eligible, THE System SHALL explain why conversationally and suggest alternatives

### Requirement 7: Conversational Document Guidance

**User Story:** As a citizen, I want to understand document requirements through conversation, so that I know exactly what to prepare.

#### Acceptance Criteria

1. WHEN eligibility is confirmed, THE System SHALL naturally transition to discussing required documents
2. WHEN explaining documents, THE System SHALL describe one document at a time and ask if the citizen has it
3. WHEN a citizen doesn't have a document, THE System SHALL explain how to obtain it conversationally
4. WHEN multiple documents serve the same purpose, THE System SHALL explain alternatives
5. WHEN all documents are confirmed, THE System SHALL summarize and provide next steps

### Requirement 8: Conversational Fraud Prevention

**User Story:** As a citizen, I want to be warned about scams naturally in conversation, so that I stay safe without feeling alarmed.

#### Acceptance Criteria

1. WHEN discussing application process, THE System SHALL naturally mention that government schemes are free
2. WHEN a citizen asks about fees, THE System SHALL conversationally explain no fees are required and warn about middlemen
3. WHEN a citizen mentions being asked for money, THE System SHALL provide official helpline numbers in a supportive tone
4. WHEN fraud patterns are detected, THE System SHALL weave warnings into the conversation naturally
5. WHEN providing warnings, THE System SHALL maintain a helpful, protective tone without breaking conversation flow

### Requirement 9: Streaming Voice Output

**User Story:** As a citizen with limited literacy, I want to hear responses immediately as they're generated, so that the conversation feels natural and responsive.

#### Acceptance Criteria

1. WHEN a response is generated in Hindi, THE Polly_Service SHALL stream audio using the Aditi voice
2. WHEN a response is generated in Marathi, THE Polly_Service SHALL stream audio using an appropriate Marathi voice
3. WHEN streaming audio, THE System SHALL start playback before the complete response is generated
4. WHEN speaking, THE System SHALL use natural conversational prosody and appropriate speaking rate
5. WHEN the citizen interrupts, THE System SHALL stop speaking and listen

### Requirement 10: Multilingual Conversation Support

**User Story:** As a Maharashtra citizen, I want to converse in Hindi, Marathi, or English, so that I can use my preferred language throughout.

#### Acceptance Criteria

1. WHEN a citizen connects, THE System SHALL greet them and ask for language preference conversationally
2. WHEN language is set to Hindi, THE System SHALL conduct the entire conversation in Hindi
3. WHEN language is set to Marathi, THE System SHALL conduct the entire conversation in Marathi
4. WHEN language is set to English, THE System SHALL conduct the entire conversation in English
5. WHEN a citizen switches language mid-conversation, THE System SHALL adapt seamlessly and continue

### Requirement 11: WebSocket Connection Management

**User Story:** As a citizen, I want a stable connection throughout my conversation, so that I don't get disconnected.

#### Acceptance Criteria

1. WHEN a citizen connects, THE API_Gateway SHALL establish a WebSocket connection and create a unique connection ID
2. WHEN a connection is established, THE System SHALL create a conversation state in DynamoDB
3. WHEN a connection is active, THE System SHALL send heartbeat messages every 30 seconds
4. WHEN a connection is idle for 30 minutes, THE System SHALL gracefully close it after asking if the citizen is still there
5. WHEN a connection drops unexpectedly, THE System SHALL allow reconnection and resume the conversation

### Requirement 12: Scheme Data Management

**User Story:** As a system administrator, I want to easily update scheme information, so that citizens always receive current information in conversations.

#### Acceptance Criteria

1. WHEN scheme documents are uploaded to S3, THE System SHALL validate the document format
2. WHEN new schemes are added, THE System SHALL generate embeddings and update Embedding_Store
3. WHEN scheme details change, THE System SHALL regenerate affected embeddings within 1 hour
4. WHEN schemes expire, THE System SHALL mark them as inactive in DynamoDB
5. WHEN scheme metadata is queried, THE System SHALL return only active schemes

### Requirement 13: WebSocket API Integration

**User Story:** As a frontend developer, I want WebSocket APIs for streaming conversations, so that I can build responsive voice interfaces.

#### Acceptance Criteria

1. WHEN a client connects via WebSocket, THE API_Gateway SHALL accept the connection and return a connection ID
2. WHEN a client streams audio, THE API_Gateway SHALL forward it to the conversation handler
3. WHEN the system generates responses, THE API_Gateway SHALL stream audio back to the client
4. WHEN connection rate exceeds limits, THE API_Gateway SHALL throttle new connections
5. WHEN authentication fails, THE API_Gateway SHALL reject the connection with error details

### Requirement 14: Conversational Performance and Scalability

**User Story:** As a product manager, I want the system to handle thousands of concurrent conversations, so that we can scale nationally.

#### Acceptance Criteria

1. WHEN a citizen speaks, THE System SHALL provide streaming transcription with < 500ms latency
2. WHEN generating responses, THE System SHALL start streaming audio within 2 seconds
3. WHEN concurrent conversations reach 1000, THE Lambda_Functions SHALL auto-scale to handle load
4. WHEN DynamoDB throughput is exceeded, THE System SHALL use on-demand scaling
5. WHEN Bedrock API rate limits are approached, THE System SHALL implement exponential backoff

### Requirement 15: Security and Compliance

**User Story:** As a compliance officer, I want data protection measures for conversations, so that we comply with DPDP Act 2023.

#### Acceptance Criteria

1. WHEN conversation data is stored, THE System SHALL encrypt it at rest using AWS KMS
2. WHEN data is transmitted via WebSocket, THE System SHALL use WSS (TLS 1.3 encryption)
3. WHEN accessing AWS services, THE System SHALL use IAM roles with least privilege
4. WHEN a citizen requests data deletion, THE System SHALL remove all conversation data within 30 days
5. WHEN audit logs are generated, THE CloudWatch_Monitor SHALL retain them for 90 days

### Requirement 16: Conversational Monitoring and Observability

**User Story:** As a DevOps engineer, I want comprehensive monitoring of conversations, so that I can detect and resolve issues quickly.

#### Acceptance Criteria

1. WHEN any conversation turn occurs, THE System SHALL log conversation state and latency to CloudWatch
2. WHEN errors occur during conversation, THE System SHALL log stack traces and conversation context to CloudWatch
3. WHEN Bedrock API calls are made, THE System SHALL track token usage and conversation quality metrics
4. WHEN streaming latency exceeds 2 seconds, THE System SHALL trigger CloudWatch alarms
5. WHEN system health is queried, THE API_Gateway SHALL provide a health check endpoint

### Requirement 17: Conversational Cost Optimization

**User Story:** As a product manager, I want to minimize costs while maintaining conversation quality, so that we can serve maximum citizens within budget.

#### Acceptance Criteria

1. WHEN selecting Bedrock models, THE System SHALL use Claude 3 Haiku for simple conversation turns and Sonnet for complex reasoning
2. WHEN generating embeddings, THE System SHALL batch process documents to reduce API calls
3. WHEN storing conversation data in DynamoDB, THE System SHALL use on-demand pricing and TTL for cleanup
4. WHEN streaming audio, THE System SHALL optimize chunk sizes to balance latency and cost
5. WHEN Lambda functions execute, THE System SHALL use appropriate memory allocation to optimize cost-performance ratio

### Requirement 18: Maharashtra Scheme Coverage (MVP)

**User Story:** As a Maharashtra citizen, I want to discuss state-specific schemes conversationally, so that I can access benefits available to me.

#### Acceptance Criteria

1. THE Scheme_Repository SHALL contain at least 50 Maharashtra government schemes across agriculture, education, health, and housing categories
2. WHEN a citizen discusses agriculture needs, THE System SHALL conversationally introduce schemes like Mahatma Jyotiba Phule Shetkari Karjmukti Yojana
3. WHEN a citizen discusses education needs, THE System SHALL conversationally introduce schemes like Rajarshi Chhatrapati Shahu Maharaj Shikshan Shulkh Punarpurti Yojana
4. WHEN a citizen discusses health needs, THE System SHALL conversationally introduce schemes like Mahatma Jyotiba Phule Jan Arogya Yojana
5. WHEN a citizen discusses housing needs, THE System SHALL conversationally introduce schemes like Pradhan Mantri Awas Yojana (Maharashtra component)

### Requirement 19: Regional Deployment

**User Story:** As a compliance officer, I want all conversation data stored in Mumbai region, so that we comply with data residency requirements.

#### Acceptance Criteria

1. THE System SHALL deploy all AWS resources in ap-south-1 (Mumbai) region
2. WHEN S3 buckets are created, THE System SHALL configure them with ap-south-1 region
3. WHEN DynamoDB tables are created, THE System SHALL configure them with ap-south-1 region
4. WHEN Lambda functions are deployed, THE System SHALL deploy them in ap-south-1 region
5. WHEN data replication is needed, THE System SHALL only replicate within Indian regions


### Requirement 20: Hackathon Demonstration Readiness

**User Story:** As a hackathon participant, I want a working conversational demo, so that I can showcase the solution to judges.

#### Acceptance Criteria

1. THE System SHALL provide a web-based interface for streaming voice conversations
2. WHEN demonstrating, THE System SHALL showcase at least 3 complete conversational journeys (agriculture, education, health schemes)
3. WHEN judges test the system, THE System SHALL conduct natural conversations in Hindi and Marathi
4. WHEN showcasing innovation, THE System SHALL demonstrate streaming RAG retrieval and conversational flow control
5. WHEN presenting architecture, THE System SHALL provide WebSocket architecture diagram and cost breakdown
