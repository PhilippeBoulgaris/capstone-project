def intToStr(i):
    digits = '0123456789'
    if i == 0:
        return '='
    result = ''
    while i > 0:
        result = digits[i%10] + result
        i = i//10
    return result

print(intToStr(123))