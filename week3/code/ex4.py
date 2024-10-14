import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout)

class Form(QMainWindow):
    def button1_slot(self):
        print('button 1 clicked')

    def button2_slot(self):
        print('button 2 clicked')

    def button3_slot(self):
        print('button 3 clicked')

    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        button1 = QPushButton("button1")
        button2 = QPushButton("button2")
        button3 = QPushButton("button3")
        layout_vert = QVBoxLayout()
        layout_vert.addWidget(button1)
        layout_vert.addWidget(button2)
        layout_vert.addWidget(button3)
        widget = QWidget(self)
        widget.setLayout(layout_vert)
        self.setCentralWidget(widget)
        button1.clicked.connect(self.button1_slot)
        button2.clicked.connect(self.button2_slot)
        button3.clicked.connect(self.button3_slot)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    sys.exit(app.exec())
