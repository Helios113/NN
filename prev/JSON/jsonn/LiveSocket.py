import websocket
try:
    import thread
except ImportError:
    import _thread as thread
import time
import json
import FileHandler as fh
import getter as gt
def on_message(ws, message):
    print("hit")
    python_dict = json.loads(message)
    if python_dict['event'] == 'trade':
        output_str = [python_dict['data']['microtimestamp'],python_dict['data']['amount_str'],python_dict['data']['price_str'],python_dict['data']['type']]
        fh.FileAdder(output_str,'/pricedata.csv')
        gt.Funk(output_str)
def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    ms = json.dumps({
                        "event": "bts:subscribe",
                        "data": {
                            "channel": "live_trades_xrpeur"
                                }
                    })
    def run(*args):
        ws.send(ms)
    thread.start_new_thread(run, ())


if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://ws.bitstamp.net",
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close,
                              on_open = on_open)
    ws.run_forever()