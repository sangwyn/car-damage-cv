import urllib.request

with open('test_urls.txt', 'r') as f:
    for img_url in f:
        img_name = img_url[28:-5]
        img_path = f'./test_data/{img_name}.jpg'
        urllib.request.urlretrieve(img_url, img_path)
        print(img_path)