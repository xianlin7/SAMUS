

def keep_img_name(path):
    with open("./imgname.txt", "w") as f:
        f.write(path)

def read_img_name():
    f = open(r"./imgname.txt", "r")
    file = f.readlines()
    for each in file:
        each = each.strip('\n')
        return each