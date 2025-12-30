# -*- coding: utf-8 -*-
import re, os


def normalize_text(text, dictionary):
    normalized_text = text
    sorted_keys = sorted(dictionary.keys(), key=len, reverse=True)

    for word in sorted_keys:
        escaped_word = re.escape(word)
        # Граница слова: пробел, начало/конец строки, или другой символ, не являющийся буквой, цифрой, подчеркиванием или точкой.
        pattern = r"(?:(?<= )|^)" + escaped_word + r"(?:(?= )|$)"  # Улучшенное определение границ слов
        #normalized_text = re.sub(pattern, dictionary[word], normalized_text, flags=re.IGNORECASE)
        normalized_text = re.sub(pattern, dictionary[word], normalized_text)

    return normalized_text

def normalize(raw_text):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    text = raw_text
    if text[-1] not in ['.', '!', '?']:
        text = text + "."
    text = re.sub(r"(\d+)\.\s*([afāgbģhicčdeēījknķlļmņoprsžštuūvz:,;-])", r"\1. \2", text) # Turns 25.decembrī  into 25. decembrī to prevent splitting
    text = re.sub(r"[.…。]+\s*", ". ", text) 
    text = text.strip().replace('!\n\r', '! ').replace('!\n', '! ').replace('!\r', '! ')
    text = text.strip().replace('.\n\r', '. ').replace('.\n', '. ').replace('.\r', '. ')
    text = text.strip().replace('?\n\r', '. ').replace('?\n', '. ').replace('?\r', '. ')
    text = text.strip().replace('\n\r', ' ').replace('\n', ' ').replace('\r', ' ')
    text = text.replace("％", "%")
    text = re.sub(r"(\d+)%", r"\1 % ", text) # 235% = 235 %
    #text = re.sub(r"(\d+)\s+(\d+)[^.!?]", r"\1\2 ", text) # 13 000 = 13000 %
    #text = re.sub(r"([–,;….!?:])+", "\1", text) # replace --- as -
    text = re.sub(r"\s+\.", ". ", text) # Arrows
    text = re.sub(r"\s+\,", ", ", text) # Arrows
    text = re.sub(r" -", " - ", text) #  -\w =  - 
    text = re.sub(r"- ", " - ", text) 
    # text = re.sub(r",*\s+,", ",", text) 
    text = re.sub(r'(\d+),(\d+)', r'\1.\2 ', text) # replace comma to dot  in digitals 
    text = re.sub(r"[„”\"'“\(\)\{\}]+", ' ', text) # clear all simbols
    text = re.sub(r"([a-zA-ZāčēģīķļņošūžĀČĒĢĪĶĻŅŌŠŪŽ])\s*([,.:;!?])\s*([a-zA-ZāčēģīķļņošūžĀČĒĢĪĶĻŅŌŠŪŽ])", r"\1\2 \3", text) # замена формата буква,буква на буква,(пробел)буква
    print(text)
    # replace by dictionary
    with open(BASE_DIR+'/lv-dictionary.txt', 'r') as file:
        dictionary = {}
        for line in file.readlines():
            tedata = line.split('|')
            dictionary[tedata[0].strip()] = tedata[1].strip()
        text = normalize_text(text, dictionary)
    text = re.sub(r"(\d+):(\d+)[.-](\d+)", r"\1. nodaļas \2. - \3. rindkopa", text) # овучка глава - пукнт - 10:8-10
    text = re.sub(r"(\d+):(\d+)", r"\1. nodaļas \2. rindkopa", text) # овучка глава - пукнт - 10:8
    text = re.sub(r"(\d+)(\.*\s*)gs\.", r"\1\2 gadsimtam ", text) # 4.gs. or 4. gs.

    #text = re.sub(r"[.!?]\s*[.!?]", ". ", text) # replace dot with spaces
    text = re.sub(r"\s+", " ", text) # replace spaces
    norm_text = text.lower().strip()

    
    return norm_text

