import numpy as np
import os
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
import pandas as pd
import csv
import NaiveBayes as nb



def train_model(path):
    print("Training")

def predict(path):
    print("predicting")

class Browse(tk.Frame):


    def __init__(self, master, initialdir='', filetypes=()):
        tk.Frame.__init__(self,master)
        self.filepath = tk.StringVar()
        self.num_bins =tk.StringVar() #default
        self._initaldir = initialdir
        self._filetypes = filetypes
        self._create_widgets()
        self._display_widgets()
        self._model = -1


    def _create_widgets(self):
        self._entry_path = tk.Entry(self, textvariable=self.filepath, font=("bold", 10))
        self._entry_bins = tk.Entry(self, textvariable =self.num_bins, font = ("bold", 10))
        self._button_train = tk.Button(self, text="Build",bg="blue",fg="white", command=self.train)
        self._button_browse = tk.Button(self, text="Browse",bg="blue",fg="white", command=self.browseModel)
        self._button_predict = tk.Button(self, text="Classify",bg="blue",fg="white", command=self.predict)
        self._label=tk.Label(self, text="Naive Bayes Classifier", bg="blue", fg="black",height=3, font=("bold", 14))
        self._labelBrowseDir = tk.Label(self, text="Directory Path")
        self._labelBinNum = tk.Label(self, text="Discretization Bins")



    def _display_widgets(self):

        self._label.pack(fill='y')
        self._labelBrowseDir.pack(fill='y')
        self._entry_path.pack(fill='x', expand=True)
        self._button_browse.pack(fill='x')
        self._labelBinNum.pack(fill='x')
        self._entry_bins.pack(fill='x', expand=True)
        self._button_train.pack(fill='y',)
        self._button_predict.pack(fill='y',)





    def predict(self):
        if(self._model != -1):
            self._model.predict("test.csv","output.txt")
            answer = messagebox.askokcancel("Naive Bayes Classifier", "Classification completed, results are in output.txt. press OK to finish")
            exit(0)
        else:
            messagebox.showerror("Error","Please train the model first")

    def train(self):
         if  self.validate_files():
             if self.validate_bins():
                 self._model = nb.NaiveBayseAlgorithm(int(self._entry_bins.get()))
                 dataset = pd.read_csv('train.csv',sep=',')
                 self._model.train(dataset)
                 answer = messagebox.askokcancel("Naive Bayes Classifier", "Building classifier using train-set is done!")






    def validate_bins(self):
        input = self._entry_bins.get()
        if not input:
            self._entry_bins.focus_set()
            messagebox.showerror("Error","No bins inserted")
            return False

        try:
                input = int(input)
        except ValueError:
                messagebox.showerror("Error","Please Enter a legal number of bins")
                return False
        else:
            return True



    def validate_files(self):
        files_path = self._entry_path.get()
        if not files_path:
            messagebox.showerror("Error","Please choose a file directory")
            return False
        required_files = ["test.csv","train.csv","Structure.txt"]
        for f in required_files:
            my_file = os.path.join(self._entry_path.get(),f)
            if not os.path.exists(my_file):
                messagebox.showerror("Error","File "+f+" is missing")
                return False
            if os.stat(my_file).st_size == 0:
                messagebox.showerror("Error","File "+f+" is empty")
                return False

        return True





    def browseModel(self):

        self.filepath.set(fd.askdirectory())




if __name__ == '__main__':
    root = tk.Tk()
    labelfont = ('times', 10, 'bold')
    root.geometry("500x500")
    filetypes = (
        ('CSV File', '*.csv'),
        ("All files", "*.*")
    )

    file_browser = Browse(root, initialdir="\\",
                          filetypes=filetypes)
    file_browser.pack(fill='y')
    root.mainloop()