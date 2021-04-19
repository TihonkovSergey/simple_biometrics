import tkinter as tk
from src.ui.config import *


class StartPage(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        label = tk.Label(self, text="MAIN MENU", font=LargeFont)
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Face Detection",
                            command=lambda: controller.show_frame('FaceDetectionPage'), width=20, height=1)
        button2 = tk.Button(self, text="Face Recognition",
                            command=lambda: controller.show_frame('FaceRecognitionPage'), width=20, height=1)
        button3 = tk.Button(self, text="Face Recognition System",
                            command=lambda: controller.show_frame('FaceRecognitionSystemPage'), width=20, height=1)
        button4 = tk.Button(self, text="Visualize Test",
                            command=lambda: controller.show_frame('VisualizeTestPage'), width=20, height=1)

        button1.pack()
        button2.pack()
        button3.pack()
        button4.pack()
