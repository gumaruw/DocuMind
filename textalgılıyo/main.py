from rag_system import RAGSystem

def main():
    rag_system = RAGSystem()
    pdf_path = input("PDF dosyasının yolunu girin: ").strip().strip('"')
    if not rag_system.load_document(pdf_path):
        print("PDF yüklenemedi!")
        return

    print("\nDöküman yüklendi. Sorularınızı sorabilirsiniz.")
    print("Çıkmak için 'exit' yazın.\n")

    while True:
        question = input("\nSorunuz: ")
        if question.lower() == 'exit':
            break
        
        answer = rag_system.answer_question(question)
        print("\nCevap:", answer)

if __name__ == "__main__":
    main()