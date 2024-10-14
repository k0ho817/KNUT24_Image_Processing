import sys
from PySide6.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)
label = QLabel("Hello World!")
label = QLabel("<font color=red size=40>Hello World!</font>")
print(f"sys: {sys.argv}")
print(f"Hello {sys.argv[1]} & {sys.argv[2]}")
label.show()
app.exec()