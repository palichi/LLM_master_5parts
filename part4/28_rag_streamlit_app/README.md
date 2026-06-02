# 📚 PDF RAG 챗봇 (Streamlit)

**Session 28 실습** — LangChain + OpenAI + Chroma + Streamlit으로 만든 PDF Q&A 챗봇.

## 🎯 기능

- 📄 **PDF 여러 개 업로드** → 자동 인덱싱 (한 번만 실행, 캐시됨)
- 💬 **대화 이력 유지** — 멀티턴 가능
- 🌊 **스트리밍 응답** — 토큰 단위로 즉시 표시
- 📚 **출처 표시** — 답변의 근거 문서 + 페이지 + 미리보기
- 🛠️ **사이드바 조절** — 모델 / 청크 크기 / Top-K 실시간 변경
- 🗑️ **초기화 버튼** — 대화 / 인덱스 리셋

## 🚀 빠른 시작 (5분)

### 1) 가상환경 + 설치
```bash
cd part4/28_rag_streamlit_app
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2) API 키 설정
```bash
cp .env.example .env
# .env 파일 열고 OPENAI_API_KEY=sk-... 실제 키로 변경
```

API 키 발급: https://platform.openai.com/api-keys

### 3) 실행
```bash
streamlit run app.py
```

브라우저가 자동으로 열림 → `http://localhost:8501`

### 4) 사용
1. 좌측 사이드바에서 **PDF 업로드**
2. 인덱싱 완료 메시지 확인 (✅ 청크 N개)
3. 하단 채팅창에 **자연어로 질문**

## 📊 사이드바 옵션

| 설정 | 기본값 | 설명 |
|------|--------|------|
| **LLM 모델** | gpt-4o-mini | 가성비 1위, 한국어 양호 |
| **청크 크기** | 500 자 | 검색 정확성에 가장 영향 큼 |
| **청크 오버랩** | 50 자 | 경계 잘림 방지 |
| **검색 Top-K** | 3 | LLM에 전달할 문서 수 |

## 💡 작동 원리 (요약)

```
[인덱싱 — 1회]
  PDF → PyPDFLoader → 청킹 → OpenAI Embedding → Chroma 저장

[질의 — 매번]
  질문 → Chroma 검색(Top-K) → LCEL 체인(prompt + LLM) → 스트리밍 답변
                                                       └── 출처 표시
```

## 📦 의존성

- `streamlit>=1.30.0` — UI 프레임워크
- `langchain` + `langchain-openai` + `langchain-chroma` — RAG 파이프라인
- `chromadb` — 벡터 DB (in-memory)
- `pypdf` — PDF 파싱
- `python-dotenv` — .env 로딩

## ☁️ Streamlit Cloud 배포

1. 이 디렉토리를 **GitHub repository로 push** (.env 제외!)
2. https://share.streamlit.io 접속 → GitHub 연동
3. **New app** → repo 선택 → `app.py` 경로 지정
4. **Secrets** 에 `OPENAI_API_KEY = "sk-..."` 입력
5. Deploy → 공개 URL 발급 (예: `https://my-pdf-rag.streamlit.app`)

## ⚠️ 주의사항

- **`.env` 파일 절대 git에 push 금지** (`.gitignore`에 포함됨)
- **Chroma는 in-memory** — 앱 재시작 시 인덱스 소실 (영구 저장 원하면 `persist_directory` 추가)
- **OpenAI 비용** — PDF 인덱싱 1회(임베딩) + 매 질문(LLM) 비용 발생
  - `gpt-4o-mini`: 1000자 질문 ≈ $0.0002 (매우 저렴)
  - `text-embedding-3-small`: 100페이지 PDF ≈ $0.001

## 🔧 흔한 문제

| 증상 | 해결 |
|------|------|
| `401 Authentication Error` | `.env`의 키가 placeholder인지 확인 |
| `Quota exceeded` | OpenAI billing 충전 |
| 답변이 엉뚱함 | 청크 크기 / Top-K 조절 |
| 앱이 느림 | 인덱싱 캐시 확인 (`@st.cache_resource`) |
| 한글 깨짐 | 거의 없음, 발생 시 PDF 텍스트 추출 품질 문제 |

## 📚 확장 아이디어

- **영구 저장**: `Chroma(persist_directory="./db")` 로 변경
- **여러 컬렉션**: 도메인별로 DB 분리 (`collection_name` 활용)
- **Reranking**: BGE-Reranker 추가 (Session 29 Advanced RAG)
- **대화형 RAG**: `ConversationalRetrievalChain`으로 교체 (이전 대화 맥락 반영)
- **로컬 LLM**: OpenAI 대신 Ollama (`langchain-ollama` + 한 줄)
- **인증**: `streamlit-authenticator` 로 로그인 추가

## 📄 라이선스

본 코드는 강의 자료로 제공됩니다.

© Copyright AIDENTIFY. All rights reserved.
