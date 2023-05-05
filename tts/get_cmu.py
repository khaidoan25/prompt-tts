from tts.utils import get_cwd
from tts.process_text import text_to_sequence, cmudict, sequence_to_text
from tts.process_text.symbols import symbols

import pandas as pd


def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


if __name__ == "__main__":
    cwd = get_cwd()    
    cmu_dict = cmudict.CMUDict(cwd + "/tts/process_text/cmu_dictionary")
    
    df = pd.read_csv("lj.csv")

    length = []
    for text in df['text']:
        cmu_sequence = intersperse(
            text_to_sequence(text, ["english_cleaners"], cmu_dict),
            len(symbols)
        )
        length.append(len(cmu_sequence))
        
    print(max(length))