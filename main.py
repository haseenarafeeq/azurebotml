import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from naive_bayes_response import nbAnswer
from lstm_response import lstmAnswer

class MyRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)

        filename = parsed_url.path.strip('/')

        reqt = {'question':'','algorithm':'' }
        query_params = parse_qs(parsed_url.query)
        for name, value in query_params.items():
            reqt[name] = value[0]

        data = {}
        if(filename=='answer'):
            self.send_response(200)
            if(reqt['question']!=''):
                if(reqt['algorithm']=='LSTM'):
                    data['response'] = lstmAnswer(reqt['question'])
                else:
                    data['response'] = nbAnswer(reqt['question'])
            else:
                data['response'] = ""
        else:
            self.send_response(404)
            data['response'] = "File Not Found!" 

        resp = json.dumps(data)
        self.send_header('Content-type', 'application/json')
        self.end_headers()        
        self.wfile.write(resp.encode('utf-8'))

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, MyRequestHandler)
    print(f"Server running on http://localhost:{port}/")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
