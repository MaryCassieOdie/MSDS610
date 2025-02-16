import tkinter as tk
from tkinter import filedialog

class LabelingTool:
    def __init__(self, master):
        self.master = master
        self.master.title("Simple Labeling Tool")
        self.label = tk.Label(master, text="Label the data")
        self.label.pack()

        self.text = tk.Text(master, height=10, width=50)
        self.text.pack()

        self.label_entry = tk.Entry(master)
        self.label_entry.pack()

        self.save_button = tk.Button(master, text="Save", command=self.save_label)
        self.save_button.pack()

        self.load_data()

    def load_data(self):
        with open("sample_text.txt", "r") as file:
            self.data = file.readlines()
        self.current_index = 0
        self.display_next_data()

    def display_next_data(self):
        if self.current_index < len(self.data):
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, self.data[self.current_index])
        else:
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, "No more data to label.")

    def save_label(self):
        data = self.text.get("1.0", tk.END).strip()
        label = self.label_entry.get()
        with open("labeled_data.txt", "a") as file:
            file.write(f"{data}\t{label}\n")
        self.label_entry.delete(0, tk.END)
        self.current_index += 1
        self.display_next_data()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingTool(root)
    root.mainloop()