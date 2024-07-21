import tkinter as tk
from tkinter import Label, filedialog, ttk
from PIL import ImageTk, Image

class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()

        self.entrythingy = tk.Entry()
        self.entrythingy.pack()

        # Create the application variable.
        self.contents = tk.StringVar()
        # Set it to some value.
        self.contents.set("this is a variable")
        # Tell the entry widget to watch this variable.
        self.entrythingy["textvariable"] = self.contents

        # Define a callback for when the user hits return.
        # It prints the current value of the variable.
        self.entrythingy.bind('<Key-Return>',
                             self.print_contents)
        
        self.btn = ttk.Button(root, text='open image', command=self.open_img).pack()

    def print_contents(self, event):
        print("Hi. The current entry content is:",
              self.contents.get())
        
    def openfn(self):
        filename = filedialog.askopenfilename(title='open')
        return filename

    def open_img(self):
        x = self.openfn()
        img = Image.open(x)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(root, image=img)
        panel.image = img
        panel.pack()

root = tk.Tk()
root.geometry("500x500")
root.resizable(width=True, height=True)

myapp = App(root)
myapp.mainloop()