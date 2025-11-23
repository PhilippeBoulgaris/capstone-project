def addDigits(s):
    val = 0
    for c in s:
        val += int(c)
    return val

s="103938503"
print(addDigits(s))