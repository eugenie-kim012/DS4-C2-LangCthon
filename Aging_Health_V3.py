import os
import streamlit as st
import tiktoken  # 토큰 계산 (예: 길이 측정)에 사용
import shutil 
import pandas as pd  # CSV 파일 처리를 위해 추가
from openai import OpenAI
from dotenv import load_dotenv  # .env 파일 로드를 위함

# ------------ LangChain 임포트 ------------------------------
# 기존 임포트들을 모두 이렇게 변경하세요:

# 벡터스토어 관련 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# 체인 관련 
from langchain.chains import ConversationalRetrievalChain

# 문서 로더
from langchain_community.document_loaders import PyPDFLoader

# 텍스트 스플리터
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 메모리
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# 문서, 프롬프트
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# --------------------- API 키 및 초기 설정 ------------------------------

st.set_page_config(page_title="🧠 OECD Aging and Health 자료 검색 봇", layout="wide")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit 사이드바에서 API 키 입력받기 (.env에서 찾으면 미리 채워짐)
st_api_key_input = st.sidebar.text_input("🔑 OpenAI API Key", type="password", value=api_key)

# 최종적으로 사용할 API 키 결정 (사이드바 입력이 우선)
openai_api_key = st_api_key_input if st_api_key_input else api_key

# Langchain 및 OpenAI 라이브러리가 API 키를 환경 변수에서 찾도록 설정
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.error("⚠️ OpenAI API 키를 입력해 주세요 (사이드바 또는 .env 파일에 OPENAI_API_KEY 설정).")
    st.stop() # 키가 없으면 앱 실행 중지

st.title("🧠 Aging and Health OECD 자료 검색 봇")

# PDF 파일이 있는 폴더 경로를 지정합니다.
# 챗봇이 학습할 PDF 파일(.pdf)을 이 폴더에 넣어주세요.
# 이 경로는 Streamlit 앱이 실행되는 서버 환경의 경로입니다.
TEXT_FOLDER = "/Users/yujin.sophia.kim/Desktop/Python/aiffel/LLM/Group_Project/OECD_Aging_Report" 

# 벡터 데이터베이스가 저장될 폴더
DB_FOLDER = "./chroma_db_local_files_pdf" 

# CSV 분류 파일 경로 자동 감지
def find_csv_file():
    """CSV 파일을 자동으로 찾습니다."""
    possible_paths = [
        "./aging_health_classification.csv",  # 현재 폴더
        "./data/aging_health_classification.csv",  # data 폴더
        "/Users/choi/Desktop/programming/aiffel/Langchain-RAG/aging_health_classification.csv",  # 절대 경로
        "/Users/choi/Desktop/programming/aiffel/Langchain-RAG/data/aging_health_classification.csv",  # data 폴더 절대 경로
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            st.sidebar.success(f"✅ CSV 파일 발견: {path}")
            return path
    
    st.sidebar.error("❌ aging_health_classification.csv 파일을 찾을 수 없습니다!")
    st.sidebar.info("다음 위치 중 하나에 파일을 배치해주세요:")
    for path in possible_paths:
        st.sidebar.write(f"- {path}")
    return None

CSV_CLASSIFICATION_PATH = find_csv_file()

# --- CSV 분류 데이터 로드 함수 ---
@st.cache_data
def load_classification_data():
    """
    CSV 분류 파일을 로드하고 전처리합니다.
    """
    try:
        df = pd.read_csv(CSV_CLASSIFICATION_PATH, encoding='utf-8')
        
        # 컬럼명 정리 (공백 제거)
        df.columns = df.columns.str.strip()
        
        # 결측값과 빈 문자열 처리
        df = df.fillna('')
        
        # 연도 컬럼 정리 - 빈 문자열을 NaN으로 변경
        if '연도' in df.columns:
            df['연도'] = df['연도'].replace('', pd.NA)
        
        # 주제 컬럼에서 복수 주제 분리 (쉼표로 구분된 경우)
        topics_expanded = []
        if '주제' in df.columns:
            for _, row in df.iterrows():
                topics_str = str(row['주제'])
                if topics_str and topics_str != '' and topics_str != 'nan':
                    # 쉼표로 분리된 주제들을 개별 항목으로 처리
                    topics = [topic.strip() for topic in topics_str.split(',')]
                    for topic in topics:
                        if topic and topic != 'nan':  # 빈 문자열이 아닌 경우만
                            topics_expanded.append(topic)

        # 국가 컬럼에서 복수 국가 분리 (쉼표로 구분된 경우)
        countries_expanded = []
        if '국가' in df.columns:
            for _, row in df.iterrows():
                countries_str = str(row['국가'])
                if countries_str and countries_str != '' and countries_str != 'nan':
                    # 쉼표로 분리된 국가들을 개별 항목으로 처리
                    countries = [country.strip() for country in countries_str.split(',')]
                    for country in countries:
                        if country and country != 'nan':  # 빈 문자열이 아닌 경우만
                            countries_expanded.append(country)
        
        return df, list(set(topics_expanded)), list(set(countries_expanded))  # 중복 제거된 고유 리스트 반환
    except Exception as e:
        st.error(f"CSV 분류 파일 로드 실패: {e}")
        return pd.DataFrame(), [], []

# --- 필터링된 파일 목록 가져오기 함수 ---
def get_filtered_files(df, selected_topics, selected_countries, selected_years):
    """
    선택된 필터 조건에 따라 파일 목록을 필터링합니다.
    """
    if df.empty:
        return []
    
    filtered_df = df.copy()
    
    # 주제 필터링 (복수 주제가 포함된 경우도 고려)
    if selected_topics and '주제' in df.columns:  # 컬럼 존재 확인
        topic_mask = filtered_df['주제'].apply(
            lambda x: any(topic in str(x) for topic in selected_topics) if pd.notna(x) else False
        )
        filtered_df = filtered_df[topic_mask]
    
    # 국가 필터링 (복수 국가가 포함된 경우도 고려)
    if selected_countries and '국가' in df.columns:  # 컬럼 존재 확인
        country_mask = filtered_df['국가'].apply(
            lambda x: any(country in str(x) for country in selected_countries) if pd.notna(x) else False
        )
        filtered_df = filtered_df[country_mask]
    
    # 연도 필터링 - 안전한 연도 비교
    if selected_years and '연도' in df.columns:  # 컬럼 존재 확인
        def is_year_match(year_value, selected_years):
            """연도 값이 선택된 연도 목록에 포함되는지 안전하게 확인"""
            try:
                if pd.notna(year_value) and str(year_value).strip() != '':
                    year_str = str(int(float(year_value)))
                    return year_str in selected_years
                return False
            except (ValueError, TypeError):
                return False
        
        year_mask = filtered_df['연도'].apply(lambda x: is_year_match(x, selected_years))
        filtered_df = filtered_df[year_mask]
    
    # 파일명 컬럼 확인 - 여러 가능성을 체크
    possible_filename_columns = ['파일명', 'filename', 'file_name', '파일이름', 'File', 'FileName']
    
    filename_column = None
    for col in possible_filename_columns:
        # 정확한 매치
        if col in df.columns:
            filename_column = col
            break
        # 공백 제거 후 매치
        stripped_columns = [c.strip() for c in df.columns]
        if col in stripped_columns:
            # 원본 컬럼명 찾기
            for orig_col in df.columns:
                if orig_col.strip() == col:
                    filename_column = orig_col
                    break
            break
    
    if filename_column:
        return filtered_df[filename_column].tolist()
    else:
        # 파일명 컬럼이 없는 경우 모든 컬럼명을 보여주고 첫 번째 컬럼을 사용
        st.sidebar.error("❌ 파일명 컬럼을 찾을 수 없습니다!")
        st.sidebar.error(f"사용 가능한 컬럼: {list(df.columns)}")
        
        if len(df.columns) > 0:
            first_col = df.columns[0]
            st.sidebar.warning(f"⚠️ 첫 번째 컬럼 '{first_col}'을 파일명으로 사용합니다.")
            return filtered_df[first_col].tolist()
        else:
            st.sidebar.error("CSV 파일에 컬럼이 없습니다!")
            return []

# --- 폴더에서 파일 읽기 함수 (필터링 적용) ---
def read_files_from_folder(folder_path, filtered_filenames=None, num_files=50): 
    """
    지정된 폴더에서 필터링된 파일들만 읽어와 (파일 경로, 파일명) 튜플 리스트를 반환합니다.
    
    Args:
        folder_path (str): 파일을 읽어올 폴더의 경로입니다.
        filtered_filenames (list): 필터링된 파일명 리스트 (None이면 모든 파일)
        num_files (int): 읽어올 최대 파일 개수입니다. 기본값은 50입니다.
    """
    file_data_list = []
    
    if not os.path.isdir(folder_path):
        st.error(f"오류: '{folder_path}' 경로를 찾을 수 없거나 디렉토리가 아닙니다.")
        return file_data_list

    # 폴더 내 모든 PDF 파일 목록을 가져옵니다.
    all_files_in_folder = [
        f for f in os.listdir(folder_path) 
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith('.pdf')
    ]
    
    # 필터링이 적용된 경우, 해당 파일들만 선택
    if filtered_filenames is not None:
        files_to_process = [f for f in all_files_in_folder if f in filtered_filenames]
    else:
        files_to_process = all_files_in_folder
    
    # 지정된 개수만큼의 파일만 처리합니다.
    files_to_read = files_to_process[:num_files]

    # 선택된 각 파일에 대해 반복하여 경로를 리스트에 추가합니다.
    for filename_only in files_to_read:
        file_path = os.path.join(folder_path, filename_only)
        file_data_list.append((file_path, filename_only))
            
    return file_data_list

# --- PDF 파일 로드 및 Langchain Document 객체로 변환 함수 (수정됨) ---
@st.cache_resource(show_spinner=False)
def load_and_prepare_documents(pdf_folder, openai_api_key, filtered_filenames=None):
    """
    지정된 폴더에서 필터링된 PDF 파일을 읽어와 Langchain Document 객체로 변환하고,
    Chroma 벡터 데이터베이스를 생성/로드합니다.
    """
    # 필터링된 파일 경로를 가져오기 위해 read_files_from_folder 사용
    raw_files_data = read_files_from_folder(pdf_folder, filtered_filenames, num_files=50) 
    
    if not raw_files_data:
        st.warning("선택된 필터 조건에 해당하는 PDF 파일이 없습니다.")
        return None, None
    
    docs = []
    for file_path, filename_only in raw_files_data:
        try:
            # extract_images 옵션을 False로 변경하여 이미지 추출 오류 방지
            loader = PyPDFLoader(file_path, extract_images=False) 
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = filename_only # 원본 파일명을 메타데이터로 추가
            docs.extend(loaded_docs)
        except Exception as e:
            st.warning(f"경고: '{filename_only}' PDF 파일을 로드하는 데 실패했습니다. 오류: {e}")
            
    if not docs:
        st.error("선택된 조건에 해당하는 문서를 로드할 수 없습니다.")
        return None, None

    # 문서 청크(분할)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    if not chunks:
        st.warning("문서 청크를 생성하지 못했습니다. 벡터 DB를 생성할 수 없습니다.")
        return None, None

    # 필터링이 적용된 경우 DB 폴더 이름에 해시 추가하여 캐시 구분
    filter_hash = hash(str(sorted(filtered_filenames)) if filtered_filenames else "all_files")
    db_folder_with_filter = f"{DB_FOLDER}_{filter_hash}"
    
    # 현재 문서로 새로운 DB 생성을 위해 기존 DB 폴더 삭제
    if os.path.exists(db_folder_with_filter):
        st.info(f"기존 벡터 DB를 삭제하고 새로운 DB를 생성합니다.")
        try:
            shutil.rmtree(db_folder_with_filter)
        except Exception as e:
            st.error(f"기존 DB 폴더를 삭제하는 데 실패했습니다: {e}")
            return None, None

    try:
        # 임베딩 생성 및 Chroma DB에 저장
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma.from_documents(chunks, embeddings, persist_directory=db_folder_with_filter)
        db.persist() # DB를 디스크에 저장
        st.success(f"새로운 벡터 DB를 성공적으로 생성했습니다. (처리된 파일: {len(raw_files_data)}개)")
        return docs, db
    except Exception as e:
        st.error(f"벡터 DB 생성 중 오류 발생: {e}. OpenAI API 키가 유효한지 확인해주세요.")
        return None, None

# CSV 분류 데이터 로드
if CSV_CLASSIFICATION_PATH and os.path.exists(CSV_CLASSIFICATION_PATH):
    classification_df, all_topics, all_countries = load_classification_data()
else:
    st.sidebar.error("📁 CSV 분류 파일을 찾을 수 없습니다!")
    uploaded_csv = st.sidebar.file_uploader(
        "CSV 분류 파일을 업로드하세요", 
        type=['csv'],
        help="aging_health_classification.csv 파일을 업로드해주세요"
    )
    
    if uploaded_csv is not None:
        # 업로드된 파일을 임시로 저장
        with open("temp_classification.csv", "wb") as f:
            f.write(uploaded_csv.getbuffer())
        CSV_CLASSIFICATION_PATH = "temp_classification.csv"
        classification_df, all_topics, all_countries = load_classification_data()
    else:
        classification_df, all_topics, all_countries = pd.DataFrame(), [], []

# ------------------ 사이드바 폼 및 필터 -------------------
with st.sidebar:
    st.header("📊 자료 필터링")
    
    # 필터 섹션
    if not classification_df.empty:
        # 주제 필터 (멀티셀렉트) - 기본값 제거
        if '주제' in classification_df.columns and all_topics:
            unique_topics = sorted(all_topics)  # 오름차순 정렬
            selected_topics = st.multiselect(
                "🎯 주제 선택", 
                options=unique_topics,
                default=[],  # 기본값 제거
                help="여러 주제를 선택할 수 있습니다."
            )
        else:
            st.warning("CSV 파일에 '주제' 컬럼이 없거나 데이터가 없습니다.")
            selected_topics = []
        
        # 국가 필터 (멀티셀렉트) - 주제와 동일한 방식으로 처리
        if '국가' in classification_df.columns and all_countries:
            unique_countries = sorted(all_countries)  # 오름차순 정렬
            selected_countries = st.multiselect(
                "🌍 국가 선택",
                options=unique_countries,
                default=[],  # 기본값 제거
                help="여러 국가를 선택할 수 있습니다. (쉼표로 구분된 복수 국가 지원)"
            )
        else:
            st.warning("CSV 파일에 '국가' 컬럼이 없거나 데이터가 없습니다.")
            selected_countries = []
        
        # 연도 필터 (멀티셀렉트) - 빈 값과 문자열 처리 개선
        if '연도' in classification_df.columns:
            def safe_convert_year(year):
                """연도를 안전하게 정수로 변환"""
                try:
                    if pd.notna(year) and str(year).strip() != '':
                        return int(float(year))  # float을 거쳐서 변환 (소수점 있을 수 있음)
                    return None
                except (ValueError, TypeError):
                    return None
            
            valid_years = []
            for year in classification_df['연도'].unique():
                converted_year = safe_convert_year(year)
                if converted_year is not None:
                    valid_years.append(str(converted_year))
            
            unique_years = sorted(valid_years, reverse=True)  # 내림차순 정렬 (최신년도 먼저)
            selected_years = st.multiselect(
                "📅 연도 선택",
                options=unique_years,
                default=[],  # 기본값 제거
                help="특정 연도의 자료를 찾고 싶을 때 선택하세요."
            )
        else:
            st.warning("CSV 파일에 '연도' 컬럼이 없습니다.")
            selected_years = []
        
        # 언어 필터 제거됨
        
        # 필터 적용 결과 표시
        filtered_files = get_filtered_files(classification_df, selected_topics, selected_countries, selected_years)
        
        # 필터가 하나도 선택되지 않았을 때 안내 메시지
        if not any([selected_topics, selected_countries, selected_years]):
            st.info("🔍 필터를 선택하면 해당 조건에 맞는 파일들이 표시됩니다.")
            st.info(f"📂 전체 파일 수: {len(classification_df)}개")
            filtered_files = []  # 아무것도 선택하지 않으면 빈 리스트
        else:
            st.info(f"🔍 필터 결과: {len(filtered_files)}개 파일")
        
        # 선택된 파일 목록 표시 (접기/펼치기)
        with st.expander("📋 선택된 파일 목록 보기"):
            if filtered_files:
                for i, filename in enumerate(filtered_files[:10], 1):  # 처음 10개만 표시
                    st.write(f"{i}. {filename}")
                if len(filtered_files) > 10:
                    st.write(f"... 외 {len(filtered_files) - 10}개 더")
            else:
                st.write("선택된 파일이 없습니다.")
    else:
        st.error("CSV 분류 파일을 로드할 수 없습니다.")
        filtered_files = None
        
    # 간소화된 정책 봇 폼 (목적과 당신은 누구인가요만 남김)
    with st.form("Againg and health 정책 봇"):
        st.subheader("초고령 시대 정책 검색 (Aging and Health)✨")

        goal = st.text_area("검색 목적", placeholder="이 검색의 목적은 무엇인가요?", key="sidebar_goal")
        method = st.selectbox("당신은 누구인가요", ["정책입법자", "학계", "시민단체", "행정가", "기타"], key="sidebar_method")
        submitted = st.form_submit_button("✍️ 관련 정책 찾아보기")

# --- 메인 콘텐츠 영역 ---

# 채팅 메시지 기록 초기화 (세션 상태에 저장됨)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "policy analyst", "content": "💡초고령 시대 보건복지 정책 수립 우리 같이 고민해 봐요💡"}]

# 문서 로드 및 DB 준비 (필터링 적용)
if not any([selected_topics, selected_countries, selected_years]):
    # 필터가 선택되지 않은 경우
    st.info("💡 위에서 필터를 선택하면 해당 문서들로만 챗봇이 학습됩니다!")
    documents, db = None, None
else:
    # 필터가 선택된 경우에만 문서 로드
    with st.spinner(f"선택된 {len(filtered_files)}개 PDF 문서 로드 및 벡터 DB 준비 중..."):
        documents, db = load_and_prepare_documents(TEXT_FOLDER, openai_api_key, filtered_files)

# 간소화된 프로포절 생성 로직 (사이드바 폼 제출 시 트리거됨)
if submitted and openai_api_key:
    if not all([goal, method]): 
        st.warning("모든 필드를 채워주세요!")
    else:
        with st.spinner("정책 분석 초안 작성 중입니다. 챗봇은 브레인스토밍 단계에서 사용하시고 추가 연구 조사를 직접 수행하시길 바랍니다."):
            system_prompt_for_proposal_generation = """
You are a senior health policy analyst, responding in a formal and professional tone. Your role is to analyze provided OECD health reports on aging and health, which have been stored in the database, in response to questions from policy makers, university professors, administrators, or citizens seeking consultation. When answering questions, prioritize information verified in the just uploaded documents only. If there are conflicts between documents, clearly state the most reliable source and briefly explain why it is considered more credible. When quoting, provide the exact sentence, the report name, and the page number.
If relevant information cannot be found in the provided reports, state clearly that there is insufficient information in the given context. If external searches are used, always disclose the source. Always aim to be helpful, concise, and directly answer the user’s question. Respond in the language of the input (English or Korean).
You should always consider who the reader is (e.g., a policymaker, professor, administrator, or citizen) when providing your consultancy, and tailor your tone and level of detail accordingly.
"""
            # 필터 정보도 프롬프트에 포함
            filter_info = f"""
선택된 필터 조건:
- 주제: {', '.join(selected_topics) if selected_topics else '전체'}
- 국가: {', '.join(selected_countries) if selected_countries else '전체'}
- 연도: {', '.join(selected_years) if selected_years else '전체'}
- 분석 대상 파일 수: {len(filtered_files) if filtered_files else 0}개
"""
            
            user_prompt_for_proposal_generation = f"""
{filter_info}

연구 설정:
- 검색 목적: {goal}
- 당신은 누구인가요: {method}
"""
            try:
                client_openai = OpenAI(api_key=openai_api_key)
                response = client_openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt_for_proposal_generation},
                        {"role": "user", "content": user_prompt_for_proposal_generation}
                    ],
                    temperature=0.7
                )
                result = response.choices[0].message.content
                st.success("✅ 정책 분석 초안이 생성되었습니다!")
                st.markdown(result)

                st.session_state['current_proposal_context'] = {
                    "goal": goal, "method": method,
                    "full_text": result,
                    "filter_info": filter_info
                }

                st.download_button(
                    label="다운로드 (정책 분석 초안)",
                    data=result.encode('utf-8'),
                    file_name="정책_분석_초안.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ 정책 분석 초안 생성 중 오류가 발생했습니다: {e}")

# 채팅 메시지 표시
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 채팅 입력 및 처리
if prompt := st.chat_input("질문이 있으시면 여기에 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 필터가 선택되지 않은 경우 안내
    if not any([selected_topics, selected_countries, selected_years]):
        msg = "🔍 먼저 사이드바에서 필터를 선택해주세요! 선택한 문서들로만 답변을 드릴 수 있습니다."
        st.session_state.messages.append({"role": "policy analyst", "content": msg})
        st.chat_message("policy analyst").write(msg)
    elif not filtered_files:
        msg = "😅 선택한 필터 조건에 해당하는 문서가 없습니다. 다른 조건을 선택해보세요!"
        st.session_state.messages.append({"role": "policy analyst", "content": msg})
        st.chat_message("policy analyst").write(msg)
    elif openai_api_key and db:
        try:
            # ConversationalRetrievalChain을 위한 채팅 기록 초기화
            msgs = StreamlitChatMessageHistory(key="chat_messages_history")
            memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True, output_key="answer")

            # 간소화된 프로포절 컨텍스트를 채팅 시스템 프롬프트와 결합
            chat_system_role_template_str = f"""
            당신은 고도로 숙련되고 전문화된 정책 분석가입니다. 당신의 주요 역할은 제공된 OECD 정책 보고서(PDF 문서) 및 생성된 정책 분석 초안을 기반으로 정보와 통찰력을 제공하여 학생들의 정책 연구를 돕는 것입니다.
            
            현재 분석 대상 문서: {len(filtered_files)}개 파일
            선택된 필터: 주제({len(selected_topics)}개), 국가({len(selected_countries)}개), 연도({len(selected_years)}개)
            
            질문에 답변할 때는 제공된 문서나 현재 정책 분석 컨텍스트의 정보를 우선적으로 사용하세요.
            제공된 문서나 분석에서 관련 정보를 찾을 수 없는 경우, 주어진 컨텍스트에서 충분한 정보가 없다고 명시하세요.
            항상 도움이 되고, 간결하며, 질문자의 질문에 직접적으로 답변하는 것을 목표로 하세요.
            """
            
            # 현재 정책 분석 컨텍스트가 있다면 시스템 프롬프트에 추가
            if 'current_proposal_context' in st.session_state and st.session_state['current_proposal_context']['full_text']:
                current_proposal_data = st.session_state['current_proposal_context']
                form_summary = (
                    f"학생의 연구 설정:\n"
                    f"- 검색 목적: {current_proposal_data.get('goal', 'N/A')}\n"
                    f"- 당신은 누구인가요: {current_proposal_data.get('method', 'N/A')}\n\n"
                    f"{current_proposal_data.get('filter_info', '')}\n"
                )
                full_proposal_text = current_proposal_data['full_text']
                
                chat_system_role_template_str += (
                    f"\n\n현재 학생과 논의 중인 정책 분석의 컨텍스트는 다음과 같습니다:\n\n"
                    f"{form_summary}"
                    f"--- 생성된 정책 분석 초안 ---\n"
                    f"{full_proposal_text}\n"
                    f"-------------------------------\n\n"
                )
            
            # 검색된 문서 컨텍스트와 질문을 포함하는 최종 프롬프트 템플릿 구성
            final_qa_template = chat_system_role_template_str + """
다음은 검색된 문서 컨텍스트입니다:
{context}

다음은 이전 대화 기록입니다:
{chat_history}

질문: {question}
답변:
"""
            
            # PromptTemplate 객체로 변환
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question", "chat_history"], 
                template=final_qa_template,
            )

            # ConversationalRetrievalChain 생성
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(),
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
                output_key="answer"
            )

            with st.spinner("답변 생성 중..."):
                # 사용자 질문으로 체인 호출
                chat_response = qa_chain.invoke({"question": prompt, "chat_history": msgs.messages})
                msg = chat_response["answer"]

                # 참고 자료 표시
                if "source_documents" in chat_response and chat_response["source_documents"]:
                    unique_sources = {}
                    for doc in chat_response["source_documents"]:
                        source_filename = doc.metadata.get("source", "알 수 없는 파일")
                        page_number = doc.metadata.get("page", None)
                        
                        if source_filename not in unique_sources:
                            unique_sources[source_filename] = set()
                        if page_number is not None:
                            unique_sources[source_filename].add(page_number)
                    
                    source_list = []
                    for filename, pages in unique_sources.items():
                        if pages:
                            sorted_pages = sorted(list(pages))
                            source_list.append(f"{filename} (페이지: {', '.join(map(str, sorted_pages))})")
                        else:
                            source_list.append(filename)
                    
                    if source_list:
                        sources_text = "참고 자료: " + "; ".join(source_list)
                        st.info(sources_text)
                    else:
                        st.info("참고 자료를 찾을 수 없습니다.")
                        
        except Exception as e:
            msg = f"채팅 중 오류가 발생했습니다: {e}"
            print(f"채팅 오류: {e}")
    else:
        msg = "API 키가 설정되지 않았거나 PDF 문서가 로드되지 않아 채팅 기능을 사용할 수 없습니다."

    st.session_state.messages.append({"role": "policy analyst", "content": msg})
    st.chat_message("policy analyst").write(msg)