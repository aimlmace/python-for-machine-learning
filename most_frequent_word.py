from collections import Counter
import re

fn=input("Enter name of file")

with open(fn,'r') as file:
    text = file.read().lower()
    
words = re.findall(r'\b\w+\b',text)
words_count = Counter(words)
common = words_count.most_common(n)
max = max(words_count.values())

for word,count in common:
    if count == max:
        print(f"{word}:{count}")