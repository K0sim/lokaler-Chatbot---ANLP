import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from PyQt6 import QtWidgets
from chat_ui import Ui_MainWindow 
from pipeline.retriever import HybridRetriever
from pipeline.generator import AnswerGenerator
from pipeline.evaluator import LLMJudge
from config.settings import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME


class ChatBotApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initialisiere Retriever und Generator
        self.retriever = HybridRetriever(DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME)
        self.generator = AnswerGenerator()
        self.judge = LLMJudge()


        # Styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QTextEdit, QTextBrowser {
                font-size: 14px;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 8px;
                background-color: #ffffff;
            }
            QPushButton {
                font-size: 14px;
                padding: 8px;
                background-color: #2d89ef;
                color: white;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1e5fad;
            }
        """)

        # Event: Button-Klick
        self.ui.pushButton.clicked.connect(self.handle_prompt)

    def handle_prompt(self):
        question = self.ui.textEdit.toPlainText().strip()
        if not question:
            self.ui.textBrowser.setText("❗ Bitte gib eine Frage ein.")
            return

        self.ui.textBrowser.setText("🔎 Antwort wird generiert… bitte warten.")

        # 1. Kontext holen
        contexts, metadaten = self.retriever.retrieve_context(question, top_k=3)
        context_combined = "\n\n".join(contexts)

        # 2. Antwort generieren
        answer = self.generator.generate_answer(context_combined, question)

        # 3. Bewertung durchführen (richtige Reihenfolge!)
        bewertung = self.judge.evaluate(question, context_combined, answer)

        # 4. Quellen anzeigen
        sources_text = "<ul>" + "".join([
            f"<li>{m['source']} (S. {m['page']}, Titel: <i>{m['title']}</i>)</li>"
            for m in metadaten
        ]) + "</ul>"

        # 5. HTML-Ausgabe
        bewertung_html = bewertung.replace("\n", "<br>")
        self.ui.textBrowser.setHtml(f"""
            <b>🟡 Frage:</b><br>{question}<br><br>
            <b>🟢 Antwort:</b><br>{answer}<br><br>
            <b>📚 Quellen:</b>{sources_text}<br><br>
            <b>⚖️ Bewertung:</b><br>{bewertung_html}
        """)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChatBotApp()
    window.show()
    sys.exit(app.exec())
