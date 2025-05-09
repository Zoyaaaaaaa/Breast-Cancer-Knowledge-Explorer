# 🎗️ Breast Cancer Knowledge Explorer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-orange?style=for-the-badge&logo=chainlink)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-brightgreen?style=for-the-badge&logo=neo4j)
![Gemini](https://img.shields.io/badge/Google_Gemini-AI-red?style=for-the-badge&logo=google)

**Transform clinical knowledge into actionable insights with natural language.**

*Empowering healthcare professionals to explore breast cancer research, treatments, and outcomes through intuitive interaction and powerful visualization.*

[Features](#-features) • 
[Architecture](#-system-architecture) • 
[Quick Start](#-quick-start) • 
[Documentation](#-documentation) • 
[Performance](#-performance-metrics)
</div>

---

## 📋 Overview

The **Breast Cancer Knowledge Explorer** bridges the gap between complex medical literature and practical clinical insights. By leveraging advanced natural language processing and graph database technology, our platform enables healthcare professionals to:

- Query clinical data using everyday language
- Visualize relationships between symptoms, diagnoses, treatments, and outcomes
- Extract structured information from unstructured medical texts
- Track evidence back to source documents for credibility and verification

Whether you're a researcher exploring treatment efficacy, an oncologist comparing therapy options, or an educator seeking to understand symptom patterns, this tool transforms how you interact with breast cancer knowledge.

## ✨ Features

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Natural Language Interface</h3>
      <p>Ask complex clinical questions in plain English without learning query languages</p>
    </td>
    <td width="50%">
      <h3>🕸️ Knowledge Graph Engine</h3>
      <p>Powered by Neo4j for intelligent relationship mapping and knowledge discovery</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📄 Intelligent Document Processing</h3>
      <p>Automatically extract entities and relationships from clinical literature</p>
    </td>
    <td width="50%">
      <h3>📊 Interactive Visualizations</h3>
      <p>Explore connections and patterns through dynamic graph visualizations</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🔍 Evidence-Based Answers</h3>
      <p>Every insight linked back to source documents with proper citations</p>
    </td>
    <td width="50%">
      <h3>🔄 Continuous Learning</h3>
      <p>Knowledge base expands with each document processed</p>
    </td>
  </tr>
</table>

## 🏗️ System Architecture

![System Architecture Diagram](https://github.com/user-attachments/assets/5582611e-392c-48ea-9dd7-71792df606ec)

### Three-Layer Architecture

1. **User Interface Layer**: Streamlit-powered web interface for document upload and natural language queries
2. **Intelligence Layer**: LangChain orchestration with Google Gemini for understanding and processing
3. **Data Layer**: Neo4j graph database storing the knowledge network

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j database instance (local or cloud)
- Google API Key with Gemini access


## 📊 How It Works

### Knowledge Graph Model

```mermaid
graph LR
    Patient[Patient] --> |HAS_SYMPTOM| Symptom[Symptom]
    Patient --> |DIAGNOSED_WITH| Diagnosis[Diagnosis]
    Diagnosis --> |SUB_TYPE| DiagnosisType[Diagnosis Type]
    Diagnosis --> |TREATED_WITH| Treatment[Treatment]
    Treatment --> |USES| Medication[Medication]
    Treatment --> |INCLUDES| Procedure[Procedure]
    Treatment --> |RESULTS_IN| Outcome[Outcome]
    Medication --> |HAS_SIDE_EFFECT| SideEffect[Side Effect]
    Procedure --> |HAS_COMPLICATION| Complication[Complication]
    
    classDef patient fill:#f9f,stroke:#333,stroke-width:2px
    classDef diagnosis fill:#bbf,stroke:#333,stroke-width:1px
    classDef treatment fill:#bfb,stroke:#333,stroke-width:1px
    classDef outcome fill:#ffb,stroke:#333,stroke-width:1px
    
    class Patient patient
    class Diagnosis,DiagnosisType diagnosis
    class Treatment,Medication,Procedure treatment
    class Outcome,SideEffect,Complication outcome
```

### Document Processing Pipeline

```mermaid
flowchart LR
    Upload[Upload Clinical Document] --> Split[Text Splitting]
    Split --> Chunk[Semantic Chunking]
    Chunk --> Entity[Entity Extraction]
    Entity --> Relation[Relationship Detection]
    Relation --> Validation[Semantic Validation]
    Validation --> GraphDB[Neo4j Integration]
    
    style Upload fill:#f9d5e5,stroke:#333,stroke-width:1px
    style Split fill:#eeeeee,stroke:#333,stroke-width:1px
    style Chunk fill:#eeeeee,stroke:#333,stroke-width:1px
    style Entity fill:#d5e8f9,stroke:#333,stroke-width:1px
    style Relation fill:#d5e8f9,stroke:#333,stroke-width:1px
    style Validation fill:#e8f9d5,stroke:#333,stroke-width:1px
    style GraphDB fill:#f9e8d5,stroke:#333,stroke-width:1px
```

### Natural Language Query Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Web Interface
    participant LLM as Gemini AI
    participant Chain as LangChain
    participant DB as Neo4j Database
    
    User->>UI: Enter question
    UI->>LLM: Process question
    LLM->>Chain: Generate Cypher query
    Chain->>DB: Execute query
    DB-->>Chain: Return results
    Chain->>LLM: Structure answer
    LLM-->>UI: Format with citations
    UI-->>User: Display answer & visualizations
```

## 📚 Documentation

### Example Queries

<table>
  <tr>
    <th>Query Type</th>
    <th>Example Question</th>
  </tr>
  <tr>
    <td><strong>Symptom Analysis</strong></td>
    <td>"What are the early warning signs of inflammatory breast cancer?"</td>
  </tr>
  <tr>
    <td><strong>Treatment Comparison</strong></td>
    <td>"Compare survival rates between mastectomy and lumpectomy for stage II patients"</td>
  </tr>
  <tr>
    <td><strong>Medication Insights</strong></td>
    <td>"Which hormone therapies have the fewest side effects for postmenopausal women?"</td>
  </tr>
  <tr>
    <td><strong>Risk Assessment</strong></td>
    <td>"What factors increase recurrence risk after initial treatment?"</td>
  </tr>
  <tr>
    <td><strong>Complication Analysis</strong></td>
    <td>"What are common complications following lymph node removal?"</td>
  </tr>
</table>

### Supported Data Sources

- Clinical research papers and journal articles
- Treatment guidelines and medical protocols
- De-identified patient records (with proper consent)
- Clinical trial data and outcomes
- Medical textbooks and reference materials

### Key Components

| Component | Purpose | Technology |
|:---|:---|:---|
| `DocumentProcessor` | Converts documents to knowledge entities | LangChain + Gemini |
| `Neo4jConnector` | Manages database connections and operations | Neo4j Python Driver |
| `QueryEngine` | Translates natural language to database queries | LangChain GraphCypherQAChain |
| `VisualizationEngine` | Creates interactive graph visualizations | NetworkX + Pyvis |
| `CitationTracker` | Links answers to source documents | Custom tracking system |

## 📊 Performance Metrics

Our system undergoes rigorous testing to ensure accuracy in clinical knowledge extraction and query processing. Below are the latest accuracy metrics:

<table>
  <thead>
    <tr>
      <th colspan="3"><strong>Breast Cancer Cypher Query Accuracy</strong></th>
    </tr>
    <tr>
      <th>Category</th>
      <th>Question</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Diagnostic Procedures</td>
      <td>Which procedures can lead to a diagnosis of operable breast cancer?</td>
      <td>✅ True</td>
    </tr>
    <tr>
      <td>Diagnostic Procedures</td>
      <td>What diagnostic pathways exist for breast cancer detection?</td>
      <td>✅ True</td>
    </tr>
    <tr>
      <td>Surgical & Treatment</td>
      <td>What procedures might lead to a mastectomy?</td>
      <td>❌ False</td>
    </tr>
    <tr>
      <td>Testing & Results</td>
      <td>What imaging procedures are used in breast cancer diagnosis?</td>
      <td>❌ False</td>
    </tr>
    <tr>
      <td>Comprehensive</td>
      <td>Show me the complete diagnostic workflow for operable breast cancer</td>
      <td>✅ True</td>
    </tr>
    <tr>
      <td>Comprehensive</td>
      <td>What are all the possible consequences of positive margins?</td>
      <td>✅ True</td>
    </tr>
    <tr>
      <td>Statistical</td>
      <td>How many different procedures can lead to operable breast cancer diagnosis?</td>
      <td>✅ True</td>
    </tr>
    <tr>
      <td>Statistical</td>
      <td>What percentage of diagnostic pathways involve imaging?</td>
      <td>❌ False</td>
    </tr>
    <tr>
      <td>Treatment Sequences</td>
      <td>Find common sequence for treatment procedures</td>
      <td>✅ True</td>
    </tr>
    <tr>
      <td>Treatments</td>
      <td>What are the treatments used?</td>
      <td>✅ True</td>
    </tr>
    <tr>
      <td>Risk Factors</td>
      <td>List all the risk factors associated</td>
      <td>❌ False</td>
    </tr>
    <tr>
      <td>Tumor Types</td>
      <td>How many tumor types are there?</td>
      <td>✅ True</td>
    </tr>
    <tr>
      <td>Procedures</td>
      <td>Is Histopathology a procedure?</td>
      <td>✅ True</td>
    </tr>
    <tr>
      <td>Medications</td>
      <td>Total number of medications and list them</td>
      <td>✅ True</td>
    </tr>
    <tr>
      <td>Symptoms</td>
      <td>What are various symptoms?</td>
      <td>✅ True</td>
    </tr>
  </tbody>
</table>

### Summary Statistics
- **Overall Accuracy Rate**: 73% (11 correct out of 15 total queries)
- **Category Strengths**: Diagnostic Procedures (100%), Symptoms (100%), Tumor Types (100%)
- **Areas for Improvement**: Statistical Queries (50%), Risk Factors (0%)

### Performance Visualization

```mermaid
pie
    title Query Accuracy by Category
    "Correct" : 11
    "Incorrect" : 4
```

We continuously monitor and improve our system's performance through regular testing and model refinement.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the AI orchestration framework
- [Neo4j](https://neo4j.com/) for graph database technology
- [Google AI](https://ai.google/) for Gemini language model
- [Streamlit](https://streamlit.io/) for the web interface
- [Indian Cancer Society](https://www.indiancancersociety.org/) for domain expertise and resources

---

<div align="center">
  <p>Made with ❤️ for advancing breast cancer research and treatment</p>
  <p>
    <a href="https://www.indiancancersociety.org/">Learn More</a>
  </p>
</div>
