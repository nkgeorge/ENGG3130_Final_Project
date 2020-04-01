#!/usr/bin/python3

import json


i = 0
with open('Man U vs Wolves.txt','w') as fp:
    with open('c-[Post Match Thread] Manchester Utd 0 - 0 Wolves 03-26-2020.json') as f:
        for line in f:
            if 'Text' in line:
                line = line.strip()
                line = line[9:]
                line = line[:-2]
                print(line)
                fp.write(line)
                fp.write('\n')
                i=i+1
                continue
print(i)
        
  
