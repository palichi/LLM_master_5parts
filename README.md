# 거대언어모델(LLM)의 활용 및 개발 전략

> 2026학년도 K-DT 강사아카데미(전공) · 120시간(15일) 집중 과정
> 강사: **김의중** (아이덴티파이 대표) · 참고도서: 『딥러닝 개념과 활용』(미어드스페이스)

딥러닝 개념부터 실전 AI 에이전트 개발·배포까지, 현업에서 즉시 활용 가능한
**LLM 풀스택 역량**을 15일 안에 완성하는 집중 과정입니다.
매 일차마다 핵심 이론을 익힌 뒤, 그 지식을 실습으로 즉시 검증하는 구조로 설계됐습니다.

---

## 🎯 학습 목표

1. **LLM 핵심 원리** — Transformer · Attention · Position Encoding 을 수학적으로 이해
2. **모델 파인튜닝** — SFT · LoRA · DPO · GRPO 를 코드로 구현하고 도메인에 정렬
3. **RAG 시스템** — Vector / Graph / Ontology RAG (Apache AGE) 직접 구축
4. **AI 에이전트** — LangGraph · MCP · A2A · Function Calling 로 멀티 에이전트 시스템
5. **Vibe Coding** — Claude Code · Windsurf 로 복잡한 에이전트를 자연어로 빠르게 개발
6. **클라우드 배포 & 모니터링** — AWS/GCP 배포, LangSmith 로 성능 모니터링

## 👥 학습 대상

- K-디지털 훈련과정 강사로 활동을 희망하며, 최근 3년 내 관련 분야 실무 6개월 이상
- AWS · Docker 경험이 있고 클라우드 분야 강사 활동을 시작하려는 분
- Python · 딥러닝 기초 보유 + LLM API 사용 경험자
- LLM 파인튜닝 · RAG · 에이전트 개발 경험을 쌓고 싶은 분

## 📋 선수 지식

`Python 기초` · `딥러닝 개념` · `PyTorch 기초` · `HuggingFace Trainer` · `SQL` · `REST API` · `Docker 기초`

---

## ⚡ 빠른 시작

```bash
# 1) 저장소 클론
git clone https://github.com/choki0715/LLM_master_5parts.git
cd LLM_master_5parts

# 2) 환경 자동 설치 (CUDA · Ollama · VS Code 포함, 약 5~15분)
bash setup.sh

# 3) API 키 입력
cp .env.example .env
#   .env 에 OPENAI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN 채우기

# 4) Jupyter 커널 "Python (LLM)" 선택 후 노트북 실행
```

> **가벼운 설치만 원하면**: `pip install -r requirements.txt` 만 실행 (CUDA·Ollama는 별도)
> **검증**: 끝나면 [setup_check.ipynb](setup_check.ipynb) 로 25개 핵심 패키지 동작 확인

---

## 📅 5 Parts × 3일 = 15일 커리큘럼

각 Part = 1주차 = 3일치 분량 · 매일 8시간 · 이론 + 실습. **굵은 글씨** = 메인 실습 노트북.

### Part 1 — LLM 기초: 딥러닝부터 microGPT까지

| 일차 | 주제 | 노트북 |
|:-:|---|---|
| **1일차** | LLM 현황 · 딥러닝 기초 · 자연어 처리 · 순차 모델 (RNN/LSTM/Seq2Seq) | **[01_deep_learning_basics](part1/01_deep_learning_basics.ipynb)** · [02_nlp_encoding_tokenization](part1/02_nlp_encoding_tokenization.ipynb) |
| **2일차** | Transformer 아키텍처 · GPT/BERT · HuggingFace 생태계 · LangChain | [03_transformer_bert_gpt](part1/03_transformer_bert_gpt.ipynb) · [04_llm_overview_sllm](part1/04_llm_overview_sllm.ipynb) · **[05_huggingface_ecosystem](part1/05_huggingface_ecosystem.ipynb)** · **[06_langchain_practice](part1/06_langchain_practice.ipynb)** |
| **3일차** | microGPT 한국어 실습 — 243줄 GPT 직접 구현 (의존성 0, GPU 불필요) | **[07_microgpt_practice](part1/07_microgpt_practice.ipynb)** |

### Part 2 — 파인튜닝: 지식증류 · SFT · LoRA · 데이터 파이프라인

| 일차 | 주제 | 노트북 |
|:-:|---|---|
| **4일차** | 지식증류 · Scaling Laws (Kaplan/Chinchilla) · MoE (DeepSeek-V3/Mixtral) | **[08_knowledge_distillation](part2/08_knowledge_distillation.ipynb)** · [09_scaling_law](part2/09_scaling_law.ipynb) · [10_moe_deepseek](part2/10_moe_deepseek.ipynb) |
| **5일차** | SFT 일반·심화 · LoRA/PEFT · Unsloth 가속 | [11_finetuning_overview](part2/11_finetuning_overview.ipynb) · [11a_transformers_trl_basics](part2/11a_transformers_trl_basics.ipynb) · [12a_lora_peft_theory](part2/12a_lora_peft_theory.ipynb) · **[12b_lora_peft_practice](part2/12b_lora_peft_practice.ipynb)** · [12c_unsloth_finetuning](part2/12c_unsloth_finetuning.ipynb) |
| **6일차** | 고품질 SFT 데이터 구축 · TRL 학습 · Instruction Tuning · LLM-as-Judge | **[13_data_synthetic_distillation](part2/13_data_synthetic_distillation.ipynb)** · [14_sft_huggingface_trl](part2/14_sft_huggingface_trl.ipynb) · [15_continuous_learning](part2/15_continuous_learning.ipynb) · **[16_instruction_tuning](part2/16_instruction_tuning.ipynb)** · [16b_llm_as_judge](part2/16b_llm_as_judge.ipynb) |

### Part 3 — 정렬·강화학습: RLHF · DPO · GRPO · 양자화 · 서빙

| 일차 | 주제 | 노트북 |
|:-:|---|---|
| **7일차** | 양자화 (GPTQ/AWQ/GGUF/QLoRA) + RL 정렬 기초 (MDP·PPO) | **[17_rl_basics_alignment](part3/17_rl_basics_alignment.ipynb)** · [22_quantization_concepts](part3/22_quantization_concepts.ipynb) · [23_gptq_awq_qlora](part3/23_gptq_awq_qlora.ipynb) · **[24_quantization_practice](part3/24_quantization_practice.ipynb)** |
| **8일차** | DPO vs RLHF — Bradley-Terry · 한국어 DPO 학습 (Llama-3.2-3B + TRL) | [18_preference_data](part3/18_preference_data.ipynb) · **[19_dpo_practice](part3/19_dpo_practice.ipynb)** · 보조: [rlhf_to_dpo_slides.pptx](part3/rlhf_to_dpo_slides.pptx) |
| **9일차** | GRPO + DeepSeek R1 분석 + Rejection Sampling SFT + vLLM 서빙 | [20_deepseek_r1_analysis](part3/20_deepseek_r1_analysis.ipynb) · [20b_rejection_sampling_sft](part3/20b_rejection_sampling_sft.ipynb) · **[21_grpo_practice](part3/21_grpo_practice.ipynb)** · [25_vllm_serving](part3/25_vllm_serving.ipynb) |

### Part 4 — RAG: Vector · Advanced · Graph · Ontology

| 일차 | 주제 | 노트북 |
|:-:|---|---|
| **10일차** | RAG 기술 스택 (기초·벡터DB·실습·평가) + Streamlit 챗봇 배포 | [26_rag_fundamentals](part4/26_rag_fundamentals.ipynb) · [27_vector_db_comparison](part4/27_vector_db_comparison.ipynb) · **[28_rag_practice](part4/28_rag_practice.ipynb)** + [Streamlit 앱](part4/28_rag_streamlit_app/) · [32_rag_evaluation](part4/32_rag_evaluation.ipynb) |
| **11일차** | Graph RAG — Neo4j · Apache AGE · LangChain GraphRAG · 4-hop 멀티홉 | [29_advanced_rag_base](part4/29_advanced_rag_base.ipynb) · **[30_graph_rag](part4/30_graph_rag.ipynb)** + [companies.txt](part4/data/companies.txt) |
| **12일차** | Ontology RAG · 팔란티어 전략 · 월드 모델 (기업 거버넌스 + 1인 1회장 제약) | **[31_ontology_rag](part4/31_ontology_rag.ipynb)** (OntologyReasoner + OntologyWorldModel) |

### Part 5 — 에이전트: Vibe Coding · MCP · A2A · LangGraph · 통합 프로젝트

| 일차 | 주제 | 노트북 |
|:-:|---|---|
| **13일차** | Vibe Coding 개념 + Claude Code 실습 환경 구축 | **[33_vibe_coding_intro](part5/33_vibe_coding_intro.ipynb)** · [34_claude_code_agent](part5/34_claude_code_agent.ipynb) |
| **14일차** | Agentic AI — Tool Calling · MCP · A2A · LangGraph 멀티 에이전트 | [35_tool_calling_function](part5/35_tool_calling_function.ipynb) · **[36_mcp_agent](part5/36_mcp_agent.ipynb)** · **[37_a2a_protocol](part5/37_a2a_protocol.ipynb)** · [38_agent_tech_stack_langgraph](part5/38_agent_tech_stack_langgraph.ipynb) |
| **15일차** | 프로젝트 발표 & 배포 — MCP+LangGraph+A2A 통합 에이전트 (FastAPI 배포) | [39_data_pipeline_training](part5/39_data_pipeline_training.ipynb) · **[40_project_training](part5/40_project_training.ipynb)** · [41_evaluation](part5/41_evaluation.ipynb) · [42_deployment](part5/42_deployment.ipynb) |

---

## 📂 Part별 노트북 인덱스 (폴더 탐색용)

| Part | 폴더 | 노트북 수 | 주제 |
|:-:|---|:-:|---|
| **1** | [part1/](part1/) | 7 | LLM 기초 · Transformer · HF · LangChain · microGPT |
| **2** | [part2/](part2/) | 13 | 지식증류 · MoE · SFT · LoRA · 데이터 합성 · Instruction Tuning |
| **3** | [part3/](part3/) | 10 | RL 기초 · DPO · DeepSeek R1 · GRPO · 양자화 · vLLM |
| **4** | [part4/](part4/) | 7 | RAG 기초 · Advanced · Graph RAG · Ontology RAG · 평가 |
| **5** | [part5/](part5/) | 10 | Vibe Coding · Tool Calling · MCP · A2A · LangGraph · 통합 프로젝트 |

**총 47개 노트북 · 약 120시간 분량**

---

## 🎬 강의 자료 (PPT)

| 위치 | 내용 |
|---|---|
| [part4/part4_day1_PM_session28.pptx](part4/part4_day1_PM_session28.pptx) | 10일차 오후 · 28 RAG 실습 + Streamlit |
| [part4/part4_day2_AM_session29.pptx](part4/part4_day2_AM_session29.pptx) | 11일차 오전 · 29 Advanced RAG |
| [part4/part4_day2_PM_session30.pptx](part4/part4_day2_PM_session30.pptx) | 11일차 오후 · 30 Graph RAG |
| [part4/part4_day3_AM_session31.pptx](part4/part4_day3_AM_session31.pptx) | 12일차 오전 · 31 Ontology RAG (48장) |
| [part4/part4_day3_PM_session32.pptx](part4/part4_day3_PM_session32.pptx) | 12일차 오후 · 32 RAG 평가 |
| [part4/rdf_concept.pptx](part4/rdf_concept.pptx) | RDF 개념 한 장 정리 (보조) |
| [part5/part5_mcp_lecture.pptx](part5/part5_mcp_lecture.pptx) | 14일차 · MCP 강의자료 (10장) |
| [part5/part5_a2a_lecture.pptx](part5/part5_a2a_lecture.pptx) | 14일차 · A2A 강의자료 (10장) |
| [part3/rlhf_to_dpo_slides.pptx](part3/rlhf_to_dpo_slides.pptx) | 8일차 보조 · RLHF→DPO 개념 |

---

## 🛠 활용 도구

| 분류 | 도구 |
|---|---|
| **LLM API** | OpenAI (GPT-4o) · Anthropic (Claude) · Gemini · HuggingFace Inference |
| **로컬 모델** | Ollama · Llama 3.2 · vLLM · llama.cpp |
| **벡터 DB** | ChromaDB · pgvector · FAISS |
| **그래프 DB** | Neo4j · Apache AGE (PostgreSQL 확장) |
| **에이전트** | LangChain · LangGraph · MCP (FastMCP) · A2A · Anthropic SDK |
| **파인튜닝** | HuggingFace Transformers · TRL · PEFT · Unsloth · DeepSpeed |
| **양자화** | bitsandbytes · auto-gptq · autoawq · llama-cpp-python |
| **평가** | RAGAS · NLTK · ROUGE · BERTScore · LLM-as-a-Judge |
| **배포** | FastAPI · Streamlit · uvicorn · Docker |
| **컨테이너** | Docker (Apache AGE / Neo4j / Ollama) |

---

## 📊 학습 진행도 추적

각 일차 종료 후 다음을 확인:

- [ ] 해당 일차 메인 노트북(굵은 글씨) 끝까지 실행 성공
- [ ] 셀 출력 결과를 PPT/노트와 비교해 의미 파악
- [ ] 핵심 함수/클래스를 노트에 옮겨 적기
- [ ] 다음 일차의 노트북 첫 3개 셀 미리보기

> 📌 **권장 순서**: 처음 학습 시 일차 순서대로(1→15)
> **참조 학습 시**: Part 단위(폴더)로 묶어서

---

## 📚 참고 자료

- 📖 **김의중**, 『딥러닝 개념과 활용』, 미어드스페이스
- 🌐 [Anthropic MCP 공식 문서](https://modelcontextprotocol.io/)
- 🌐 [A2A Protocol Spec](https://github.com/google/a2a-protocol) (Google)
- 🌐 [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- 🌐 [HuggingFace TRL](https://huggingface.co/docs/trl/)
- 🌐 [Apache AGE](https://age.apache.org/) · [Neo4j Cypher](https://neo4j.com/docs/cypher-manual/)

---

## 🆘 자주 발생하는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `CUDA out of memory` | 배치 크기 ↓ , `gradient_checkpointing=True` , bf16 / fp16 |
| `OPENAI_API_KEY not found` | `.env` 미설정 — `cp .env.example .env` 후 키 입력 |
| `mcp` 모듈 없음 | `pip install "mcp[cli]>=1.2.0"` (또는 `bash setup.sh` 재실행) |
| Neo4j / AGE 연결 실패 | Docker 컨테이너 미실행 — 노트북 30/31 의 docker 명령 참조 |
| `unsloth` 설치 실패 | 선택 패키지 — 12c 노트북만 영향, `pip install unsloth --no-deps` |
| Korean 폰트 깨짐 | matplotlib 한글 폰트 — `fc-cache -fv` 또는 영문 라벨 사용 (본 과정 정책) |

---

## 📝 라이선스

본 교육 자료는 K-DT 강사아카데미 전공 과정용입니다.

© 2026 AIDENTIFY. All rights reserved.
