## 🧠 OECD Aging and Health 자료 검색 봇

본 프로젝트는 초고령 시대 보건복지 법안 제의 및 보고서 작성을 지원하고 정보 접근성을 강화하기 위해 기획된 AI 챗봇입니다. Streamlit과 LangChain을 기반으로 개발된 이 챗봇은 OECD에서 발간한 관련 보고서를 탐색하고, 정책 분석 초안 작성을 보조하며, 관련 질의응답을 지원합니다. 사용자는 PDF 보고서를 업로드하고, 주제/국가/연도별로 자료를 필터링하며, OpenAI API를 활용하여 질의응답 및 정책 분석을 수행할 수 있습니다. 정책 입안자, 학계, 행정가, 시민단체의 이해 관계자의 질문을 선임 보건 정책 분석가 페르소나가 초안을 작성해 주도록 작업되었습니다.

---

### 🚀 주요 기능

- 📂 **PDF 기반 문서 검색**: OECD Aging and Health 보고서 PDF를 벡터 DB로 변환하여 검색
- 🔍 **주제 / 국가 / 연도 필터링**: CSV 파일 기반으로 자료 필터링
- 🤖 **정책 질의응답 챗봇**: 선택한 문서 기반 맞춤형 Q&A 지원
- 📝 **정책 분석 초안 생성**: 간소화된 정책 프로포절 초안 작성 (OpenAI GPT API 사용)
- 💾 **분류 CSV 자동 탐지 및 업로드**

---

### 🚀 기술 스텍

| **기술 범주** | **사용된 기술 / 라이브러리** | **역할 / 기능** |
| --- | --- | --- |
| **프론트엔드 / 앱 프레임워크** | Streamlit | 웹 UI 제공 (사이드바, 챗 인터페이스, 폼, 다운로드 버튼) |
|  | Python | 애플리케이션 전체 로직 구현 |
| **데이터 엔지니어링** | pandas | CSV 데이터 로드, 전처리, 필터링 (주제, 국가, 연도) |
|  | os, shutil | 파일 입출력, 폴더 관리, DB 폴더 삭제 |
|  | dotenv | .env 파일에서 API 키 로드 |
| **AI / 자연어 처리 (NLP)** | OpenAI (GPT-4o) | 사용자 질문 답변 생성, 정책 분석 초안 생성 |
|  | LangChain | RAG 체인 생성, 문서 검색과 LLM 응답 결합 |
|  | OpenAIEmbeddings | 문서 벡터 임베딩 생성 |
|  | Chroma | 로컬 벡터 데이터베이스, 벡터 검색 |
|  | ConversationalRetrievalChain | 문서 검색 기반 대화형 QA 체인 |
|  | ConversationBufferMemory | 대화 히스토리 관리 |
|  | StreamlitChatMessageHistory | Streamlit 내 채팅 기록 연동 |
|  | PromptTemplate | LLM 프롬프트 템플릿 동적 생성 |
| **문서 처리 / 정보 검색** | PyPDFLoader (LangChain Community) | PDF 문서 텍스트 추출 |
|  | RecursiveCharacterTextSplitter | 문서 청크 분할 (RAG 검색 최적화) |
| **API / 보안 관리** | Streamlit + .env + os.environ | OpenAI API 키 보안 관리, 런타임 환경 전달 |

---

### 📁 프로젝트 폴더 구조 예시

```
bash
CopyEdit
your-repo/
 ├── Aging_Health_V3.py
 ├── requirements.txt
 ├── data/
 │    └── aging_health_classification.csv
 ├── OECD_Aging_Report/      # PDF 파일 위치
 └── chroma_db_local_files_pdf/  # 벡터 DB 저장 폴더
```

---

### 💡 사용 예시

- OECD Aging and Health 보고서 PDF를 `OECD_Aging_Report/` 폴더에 추가
- `data/aging_health_classification.csv` 파일 준비
- Streamlit 웹앱에서 주제, 국가, 연도 필터 선택

---

### ⚠️ 주의사항

- OpenAI API 키가 필요합니다.
- PDF 보고서와 CSV 분류 파일이 준비되어 있어야 챗봇이 작동합니다.
- 필터 조건에 맞는 PDF가 준비되지 않으면 챗봇이 답변을 생성하지 않습니다.

---

### 🙌 기여

해당 프로젝트는 모두의 연구소 랭체인톤 프로젝트 발표를 위해 기획된 프로젝트입니다. 기여나 개선 아이디어가 있다면 부담없이 남겨주시길 바랍니다.
