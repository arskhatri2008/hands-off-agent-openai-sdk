from openai import AsyncOpenAI
from agents import Agent, Runner, handoff,OpenAIChatCompletionsModel, RunConfig, RunContextWrapper
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("api_key")  # Ensure you have set this in your .env file
base_url = os.getenv("base_url")  # Optional, if you have a specific base URL

external_client = AsyncOpenAI(
    api_key=api_key,  # Replace with your actual API key
    base_url=base_url
)

external_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    # openai_   client=external_client,
    model=external_model,
    tracing_disabled=True,
)

NextJs_Agent = Agent(
    name="NextJs Assistant",
    instructions="You are a helpful assistant that provides information"
)

Python_Agent = Agent(
    name="Python Assistant",
    instructions="You are a helpful assistant that provides information about Python programming."
)

NextJs_handoff = handoff(
    agent=NextJs_Agent,
    on_handoff=on_handoff,
)

Triage_Agent = Agent(
    name="Triage Assistant",
    instructions="You are a helpful assistant that navigates between NextJs and Python assistants based on the user's needs.",
    handoffs=[NextJs_handoff, Python_Agent]
)

async def on_handoff(ctx: RunContextWrapper[None], ):
    print(f"Escalation agent called with reason: {ctx.reason}")

result = Runner.run_sync(Triage_Agent, "I want to help regarding NextJs routing",run_config=config)

print(result.final_output)

print(result.last_agent)