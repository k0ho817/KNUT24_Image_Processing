import sys
import cv2
from PySide6.QtWidgets import (QLineEdit, QPushButton, QApplication, QWidget, QVBoxLayout, QMainWindow)

class Form(QMainWindow):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        self.edit = QLineEdit("Write image path here")
        self.button = QPushButton("Show Image")
        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.button.clicked.connect(self.greetings)
    
    def greetings(self):
        self.image = cv2.imread(self.edit.text())
        cv2.imshow("img", self.image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    sys.exit(app.exec())
