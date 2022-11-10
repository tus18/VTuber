import os

def setting():
    f = open('./user.txt', 'w')

r = os.path.exists('user.txt')
print(r) # True (存在する)
if r == False:
    print("初期設定を行います")
    setting()
elif r == True:
    f = open('./user.tex','r')
    lins = f.read()

