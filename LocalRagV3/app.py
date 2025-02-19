import streamlit as st
import os
from rag_system import RAGSystem
import tempfile

def init_rag_system():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
        st.session_state.messages = []

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def main():
    st.set_page_config(
        page_title="DocuMind - Akıllı Doküman İşleme ve Yanıt Sistemi",
        page_icon="📚",
        layout="centered"
    )

    # Başlık ve açıklama
    st.title("📚 DocuMind - Akıllı Doküman İşleme ve Yanıt Sistemi")
    
    # Yan panel için konteyner
    with st.sidebar:
        st.header("Doküman Yükleme")
        uploaded_file = st.file_uploader(
            "PDF dosyanızı yükleyin",
            type="pdf",
            help="Soru sormak istediğiniz PDF dosyasını seçin"
        )

    # RAG sistemini başlat
    init_rag_system()

    # Doküman yükleme işlemi
    if uploaded_file and 'current_file' not in st.session_state:
        with st.spinner("Doküman yükleniyor ve işleniyor..."):
            temp_file_path = save_uploaded_file(uploaded_file)
            if st.session_state.rag_system.load_document(temp_file_path):
                st.session_state.current_file = uploaded_file.name
                st.success(f"✅ {uploaded_file.name} başarıyla yüklendi!")
                # Geçici dosyayı sil
                os.unlink(temp_file_path)
            else:
                st.error("❌ Doküman yüklenirken bir hata oluştu!")

    # Soru-cevap alanı
    if 'current_file' in st.session_state:
        st.write(f"📄 Yüklü doküman: **{st.session_state.current_file}**")
        
        # Soru input alanı
        question = st.text_input(
            "Doküman hakkında bir soru sorun:",
            placeholder="Örnek: Tablodaki en yüksek değer nedir?",
            key="question_input"
        )

        # Soru sorma butonu
        if st.button("Soru Sor", type="primary"):
            if question:
                with st.spinner("Yanıt oluşturuluyor..."):
                    # Soruyu ve cevabı kaydet
                    st.session_state.messages.append({"role": "user", "content": question})
                    answer = st.session_state.rag_system.answer_question(question)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

        # Sohget geçmişini göster
        if st.session_state.messages:
            st.write("---")
            st.subheader("Sohbet Geçmişi")
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.write(f"👤 **Soru:** {msg['content']}")
                else:
                    st.write(f"🤖 **Yanıt:** {msg['content']}")
                st.write("---")

        # Yeni doküman yükleme butonu
        if st.sidebar.button("Yeni Doküman Yükle"):
            st.session_state.clear()
            st.experimental_rerun()

    else:
        st.info("👈 Başlamak için sol panelden bir PDF dosyası yükleyin.")

if __name__ == "__main__":
    main()