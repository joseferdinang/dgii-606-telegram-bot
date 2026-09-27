import urllib.request
import urllib.parse
import http.cookiejar

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
urllib.request.install_opener(opener)

try:
    # 1. Establish session
    req_session = urllib.request.Request("http://10.0.0.1/", headers={'User-Agent': 'Mozilla/5.0'})
    with opener.open(req_session) as res:
        res.read()

    # 2. Get Rand Info
    req = urllib.request.Request(
        "http://10.0.0.1/asp/GetRandInfo.asp?&1=1",
        data=urllib.parse.urlencode({'Username': '48575443071001AC'}).encode('utf-8'),
        headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'http://10.0.0.1/'}
    )
    with opener.open(req) as res:
        print("Status:", res.getcode())
        print("Response:", res.read().decode('utf-8', errors='ignore'))
except Exception as e:
    print("Error:", e)
