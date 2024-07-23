import re
import os
import logging
from typing import Callable
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_bolt import App, Say, Ack, BoltContext
from slack_sdk import WebClient

from openai import OpenAI, OpenAIError
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
#from supabase.client import create_client
#from langchain_community.vectorstores import SupabaseVectorStore

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
#SUPABASE_URL = os.environ.get("SUPABASE_URL")
#SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

logging.basicConfig(level=logging.DEBUG)

#model_engine = "gpt-4o"
model_engine="gpt-3.5-turbo"
client = OpenAI()
embeddings = OpenAIEmbeddings()

#supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
#vector_store = SupabaseVectorStore(
#    embedding=embeddings,
#    client=supabase_client,
#    table_name="documents",
#    query_name="match_documents",
#)
vector_store = Chroma(persist_directory="data/chromadb/", embedding_function=embeddings)

conversation_history = [
            {"role": "system", "content": "You specialized AI assistant trained in SQL and Bigquery."},
            {"role": "system", "content": "Answer with with helpful, concise responses with examples"},
            {"role": "assistant", "content": "Hello, I'm here to help. How can I assist you with Bigquery?"}
]

app = App()
app_handler = SlackRequestHandler(app)

@app.command("/fast")
def hello(body, ack):
    user_id = body["user_id"]
    ack(f"Hi <@{user_id}>!")
#    respond(f"Hi is my response to <@{user_id}>!")

def ask_llm(query):
    answer = "No answer available" 
    matched_docs = vector_store.similarity_search(query, k=10)
    results = "\n".join([d.metadata['source'] + d.page_content for d in matched_docs])
    
    user_input = f"""
    Question:
    {query}
    
    Assist with answering the Bigquery question, based on the documentation and context below:
    {results}
    """
    
    try:
        conversation_history.append({"role": "user", "content": user_input})
        response = client.chat.completions.create(model=model_engine, messages=conversation_history)
        assistant_reply = response.choices[0].message.content
        answer = assistant_reply
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        error_message = f"Error: {str(e)}"
        answer = "Please try again later and error occured" 

    return answer

@app.middleware
def log_request(logger: logging.Logger, body: dict, next: Callable):
    logger.debug(body)
    return next()

#@app.message("askthedocs")
@app.command("/docs")
#def ask_docs(body, say: Say, ack: Ack):
def ask_docs(body: dict, say: Say, ack: Ack, client: WebClient):
    #print(body)
    ack("Request received. Will begin processing")
    #event = body["event"]
    #thread_ts = event.get("thread_ts", None) or event["ts"]
    #query=str(event['text']).replace("askthedocs", "")
    query = str(body['text'])
    
    say("""Processing this request. Stand by... '""" + query + """'
------------------------------------""")
    answer = ask_llm(query)
    #say("In response to this request: " +query + "\nAnswer:\n"+answer)
    client.chat_postMessage(
        channel=body['channel_id'],
        text=answer,
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": answer
                }
            }
        ]
    )

@app.message("test")
def reply_to_test(say):
    say("Yes, tests are important!")

@app.message(re.compile("bug"))
def mention_bug(say):
    say("Do you mind filing a ticket?")

# middleware function
def extract_subtype(body: dict, context: BoltContext, next: Callable):
    context["subtype"] = body.get("event", {}).get("subtype", None)
    next()

# https://api.slack.com/events/message
# Newly posted messages only
# or @app.event("message")
#@app.event({"type": "message", "subtype": None})
#def reply_in_thread(body: dict, say: Say):
#    event = body["event"]
#    thread_ts = event.get("thread_ts", None) or event["ts"]
#    say(text="Hey, what's up?", thread_ts=thread_ts)

@app.event(
    event={"type": "message", "subtype": "message_deleted"},
    matchers=[
        # Skip the deletion of messages by this listener
        lambda body: "You've deleted a message: "
        not in body["event"]["previous_message"]["text"]
    ],
)
def detect_deletion(say: Say, body: dict):
    text = body["event"]["previous_message"]["text"]
    say(f"You've deleted a message: {text}")

# https://api.slack.com/events/message/file_share
# https://api.slack.com/events/message/bot_message
@app.event(
    event={"type": "message", "subtype": re.compile("(me_message)|(file_share)")},
    middleware=[extract_subtype],
)
def file_share_event(body: dict, say: Say):
    event = body["event"]
    thread_ts = event.get("thread_ts", None) or event["ts"]
    try:
        files = body["event"]["files"]
        say(text="Hey, fileshare: "+str(files[0]['url_private_download']), thread_ts=thread_ts)
    except Exception as e:
        say(text="sorry no file upload links available, subtype was bot message", thread_ts=thread_ts)
  
# This listener handles all uncaught message events(The position in source code matters)
@app.event({"type": "message"}, middleware=[extract_subtype])
def just_ack(logger, context):
    subtype = context["subtype"]  # by extract_subtype
    logger.info(f"{subtype} is ignored")

# App Mentions
@app.event("app_mention")
def hello_command(ack, body, say, client):
    # message = str(body['event']['blocks'][0]['elements'][0]['elements'][1]['text'])
    response = "Dev Default Default" 
    try:
        message = str(body['event']['text'])
        response = "DEV Bot @ App Mention Routed this message: " +message
    except Exceptoin as e:
        print(str(e))
        response = str(e)

    say(str(response))


@app.event("message")
def handle_message():
    pass

from fastapi import FastAPI, Request

api = FastAPI()

@api.get("/")
def root():
    return {"message": "Hello World"}

@api.post("/slack/events")
async def endpoint(req: Request):
    return await app_handler.handle(req)



