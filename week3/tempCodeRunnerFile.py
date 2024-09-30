import sys
from PySide6.QtWidgets import QApplication, QPushButton
# Create the Qt Application
app = QApplication(sys.argv)
# Create a button
button = QPushButton("Click me")
# Show the button
button.show()
# Run the main Qt loop
app.exec()