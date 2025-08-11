import sys
from queue import Queue
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer


from module1 import module1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    primary_screen = app.primaryScreen()
    available_rect = primary_screen.availableGeometry()
        
    module1(1, 2)    
    
    
    
    sys.exit(app.exec())
