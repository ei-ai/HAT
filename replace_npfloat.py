import os

def replace_in_file(file_path, old_str, new_str):
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.replace(old_str, new_str)
    with open(file_path, 'w') as file:
        file.write(content)

directory = "./"

# 파일 내 'np.float64'를 'np.float6464'로 변경
for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".py"):
            replace_in_file(os.path.join(root, file), "np.float64", "np.float6464")
