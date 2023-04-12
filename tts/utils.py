import os


def get_cwd():
    while not os.getcwd().endswith("prompt-tts"):
        os.chdir("..")
        
    return os.getcwd()