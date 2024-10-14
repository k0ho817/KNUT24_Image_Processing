import sys
from PySide6.QtWidgets import (QLineEdit, QPushButton, QApplication, QWidget, QVBoxLayout, QMainWindow)

class Form(QMainWindow):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        self.edit = QLineEdit("Write my name here")
        self.button = QPushButton("Show Greeting")
        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.button.clicked.connect(self.greetings)
    
    def greetings(self):
        print(f"Hello {self.edit.text()}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    sys.exit(app.exec())
