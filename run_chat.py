from chat import get_response

while True:
    msg = input("You: ")
    if msg.lower() == "quit":
        break
    print("Bot:", get_response(msg))
