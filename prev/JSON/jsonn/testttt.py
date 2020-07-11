import sys
for line in sys.stdin:
    win = 1
    nextExpected = []
    for i in line[:-1]:
        if i == '(':
            nextExpected.append(")")
        if i == '{':
            nextExpected.append("}")
        if i == '[':
            nextExpected.append("]")
         
        if i == ')':
            if len(nextExpected)>0:
                if nextExpected[-1] == ")":
                    del nextExpected[-1]
                else:
                    win =0
                    break
            else:
                win =0
                break
        if i == '}':
            if len(nextExpected)>0:
                if nextExpected[-1] == "}":
                    del nextExpected[-1]
                else:
                    win =0
                    break
            else:
                win =0
                break
        if i == ']':
            if len(nextExpected)>0:
                if nextExpected[-1] == "]":
                    del nextExpected[-1]
                else:
                    win =0
                    break
            else:
                win =0
                break
    if len(nextExpected) > 0:
        win = 0
    
                
    if win == 1:
        print(True, end="")
    else:
        print(False, end="")