from src.ui import *
from src.ui.config import *


class PageContainer(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        container = tk.Frame(self)
        x = self.winfo_screenwidth() // 2 - APP_WIDTH // 2
        y = self.winfo_screenheight() // 2 - APP_HEIGHT // 2
        self.geometry('{}x{}+{}+{}'.format(APP_WIDTH, APP_HEIGHT, x, y))

        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frame = {}

        for F in (StartPage, FaceDetectionPage, FaceRecognitionPage, FaceRecognitionSystemPage):
            frame = F(container, self)

            self.frame[type(frame).__name__] = frame

            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame('StartPage')

    def show_frame(self, cont):
        frame = self.frame[cont]
        frame.tkraise()

