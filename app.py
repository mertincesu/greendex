__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import TextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
import requests

pdf_path = st.secrets["pdf_path"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
api_url = "https://api.openai.com/v1/chat/completions"

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {OPENAI_API_KEY}',
}

# Sayfa düzenini geniş olarak ayarlayın
st.set_page_config(layout="wide")

# Sohbet geçmişi için oturum durumunu başlatın
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.markdown('# Greendex Skor Başvuru Formu')
st.text('Şirketinizin Türkiye’deki çevresel etkisini değerlendirmek için Greendex Skoru başvurusu yapın.')

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

class ParagraphTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split('\n\n')

@st.cache_resource
def load_pdf():
    pdf_name = pdf_path
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        vectorstore_cls=Chroma, # NEREDE (Pinecone olabilir, Weavier olabilir)
        # embedding=HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L12-v2'), # NE ILE (BERT olabilir, OpenAI embedding alici olabilir en iyisi bu ama parali)
        embedding = OpenAIEmbeddings(chunk_size=15),
        text_splitter=ParagraphTextSplitter() # NELERI (Bu chunking sistemi. Ister paragraf paragraf, ister kelime sayisi olarak (200 kelime 200 kelime vs) ayir)
    ).from_loaders(loaders)
    
    st.write("PDF yüklendi ve indekslendi.")  # Hata ayıklama açıklaması
    return index

index = load_pdf()

def prompt_func(query, n):
    if n == 1:
        prompt = (
            "Greendex is a scoring system that companies fill out a form for. The score is about how environmentally friendly you are. The form that the companies have to fill out have these questions: "
            "1.1 Sirket Adi, 1.2 Sektor, 1.3 Merkez Lokasyonu, 1.4 Calisan Sayisi, 2.1 Birincil Enerji Kaynagi, 2.2 Atik Yonetim Sistemi Varligi, 2.3 Karbon Ayak Izi, 2.4 Su Kullanimi ve koruma uygulamalari, 3.1 Surdurebilirlik Projeleri, 3.2 Cevresel duzenlemelere uyumluluk"
            "Please classify the following query into one of the following categories, taking in consideration what greendex is and how the form is structured: "
            "'General Greendex Info', 'Greendex Form Specific Inquiry', 'Greeting', 'Not Understandable Word/Phrase', 'Other Topic'"
            "Examples for these intents are: 'Greendex General info: Why is Greendex Important?', 'Greendex Form Specific Inquiry: How do I answer question 3.2?', 'Not Understandable Word/Phrase: hfixnsi' and 'Other Topic: What is your favourite car model?'"
            "Your response should ONLY be one of the categories provided, with no additional words. So your ourput format is only the category from one of the provided ones, with no additonal words"
            "Query: " + query
        )
    elif n == 2:
        prompt = (
            "You are a virtual assistant chatbot that is programmed to support users that have questions about 'Greendex Basvuru Formu'."
            "Respond politely to this user input: " + query
        )

    return prompt

def openaiAPI(prompt, max_tokens=100):
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI user query classifier that is very experienced. You classify user inputs to one of the provided classes accurately"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.5,
    }
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        category = response.json()['choices'][0]['message']['content'].strip()
        return category
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_best_matching_text(llm, index, query):

    prompt = prompt_func(query, 1)
    category = openaiAPI(prompt)

    st.write(category)

    if (category == "General Greendex Info") or (category == "Greendex Form Specific Inquiry"):
        retriever = index.vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        result = qa_chain.run(query)
        if result == "Bilmiyorum":
            result = "Özür dilerim, aradığınız bilgiye şu anda sahip değilim. Size yardımcı olabileceğim başka bir konu veya soru varsa lütfen bana bildirin."
    elif category == "Greeting":
        prompt2 = prompt_func(query, 2)
        result = openaiAPI(prompt2)
    elif category == "Not Understandable Word/Phrase":
        result = "Söylediğinizi tam olarak anlayamadım, lütfen tekrar sorabilir misiniz? Size en iyi şekilde yardımcı olmak istiyorum."
    elif category == "Other":
        result = "Ne yazık ki bu konuda size yardımcı olamıyorum. Greendex Başvuru Formu ile ilgili herhangi bir sorunuz varsa, lütfen sormaktan çekinmeyin."

    return result

# Şirketler için kapsamlı başvuru formu
with st.form(key='greendex_form'):
    # Temel Şirket Bilgileri
    st.markdown("#### 1. Temel Bilgiler")
    company_name = st.text_input("1.1 - Şirket Adı")
    industry = st.selectbox("1.2 - Sektör", ["Üretim", "Teknoloji", "Perakende", "Enerji", "Diğer"])
    headquarters = st.text_input("1.3 - Merkez Lokasyonu")
    number_of_employees = st.number_input("1.4 - Çalışan Sayısı", min_value=1)

    # Çevresel Uygulamalar
    st.markdown("#### 2. Çevresel Uygulamalar")
    energy_source = st.selectbox("2.1 - Birincil Enerji Kaynağı", 
                                 ["Yenilenebilir (Güneş, Rüzgar, Hidro)", "Yenilenemez (Kömür, Petrol, Doğalgaz)", "Karışık"])
    waste_management = st.radio("2.2 - Atık yönetim sisteminiz var mı?", ["Evet", "Hayır"])
    carbon_footprint = st.text_area("2.3 - Karbon ayak izinizi azaltma çabalarınızı açıklayın")
    water_usage = st.text_area("2.4 - Su kullanımınız ve koruma uygulamalarınızı açıklayın")

    # Sürdürülebilirlik Girişimleri
    st.markdown("#### 3. Sürdürülebilirlik Girişimleri")
    sustainability_projects = st.text_area("3.1 - Şirketinizin gerçekleştirdiği herhangi bir sürdürülebilirlik projesini listeleyin")
    community_engagement = st.text_area("3.2 - Şirketinizin çevresel konularda toplulukla nasıl etkileşimde bulunduğunu açıklayın")

    # Uyumluluk ve Sertifikalar
    st.markdown("#### 4. Uyumluluk ve Sertifikalar")
    environmental_certifications = st.text_input("4.1 - Herhangi bir çevresel sertifika listeleyin (ör. ISO 14001)")
    legal_compliance = st.radio("4.2 - Şirketiniz tüm ilgili çevresel düzenlemelere uyuyor mu?", ["Evet", "Hayır"])

    # Form gönderim düğmesi
    submit_form = st.form_submit_button(label='Başvuruyu Gönder')

    if submit_form:
        st.success("Başvuru başarıyla gönderildi!")
        st.write(f"Şirket Adı: {company_name}")
        st.write(f"Sektör: {industry}")
        st.write(f"Merkez Lokasyonu: {headquarters}")
        st.write(f"Çalışan Sayısı: {number_of_employees}")
        st.write(f"Birincil Enerji Kaynağı: {energy_source}")
        st.write(f"Atık Yönetim Sistemi: {waste_management}")
        st.write(f"Karbon Ayak İzi Çabaları: {carbon_footprint}")
        st.write(f"Su Kullanımı ve Koruma: {water_usage}")
        st.write(f"Sürdürülebilirlik Projeleri: {sustainability_projects}")
        st.write(f"Topluluk Katılımı: {community_engagement}")
        st.write(f"Çevresel Sertifikalar: {environmental_certifications}")
        st.write(f"Yasal Uyumluluk: {legal_compliance}")

# Kenar çubuğu AI Asistanı
with st.sidebar:
    st.markdown("## AI Asistanı")

    # st.chat_input'u kenar çubuğunda kullanın
    ai_query = st.chat_input("Bana bir şey sor:")

    if ai_query:
            # Proceed with processing the query
            with st.spinner('AI düşünüyor...'):
                ai_response = get_best_matching_text(llm, index, ai_query)
                st.session_state.chat_history.append({"user": ai_query, "ai": ai_response})
# Kenar çubuğunda sohbet geçmişini göster
with st.sidebar:
    for chat in reversed(st.session_state.chat_history):
        st.chat_message('user').markdown(chat['user'])
        st.chat_message('assistant').markdown(chat['ai'])
