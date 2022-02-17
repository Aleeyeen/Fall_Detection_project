import requests
import time

url = 'https://notify-api.line.me/api/notify'
    #token Personal
token = 'iLm162K7YhYixQYoEKcinQbqxHOCcYKEyNEahtEkDci'
    #token Family
#token = 'iszDQGSWKf62Ot51CGDXdD2iLFxuiq8D2kuBjbye0b6'
headers = {'content-type': 'application/x-www-form-urlencoded',
           'Authorization': 'Bearer '+token}

for i in range(3):
    r = requests.post(url, headers=headers,
                      data={'message': 'Test Fall Detect System' + str(i + 1)})
    print(r.text)
    time.sleep(5)




