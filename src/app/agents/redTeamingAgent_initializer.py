# Azure imports
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy
from pyrit.prompt_target import OpenAIChatTarget
import httpx
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Azure AI Project Information
azure_ai_project = os.getenv("FOUNDRY_ENDPOINT")

# Instantiate your AI Red Teaming Agent
red_team_agent = RedTeam(
    azure_ai_project=azure_ai_project,
    credential=DefaultAzureCredential(),
    risk_categories=[
        RiskCategory.Violence,
        RiskCategory.HateUnfairness,
        RiskCategory.Sexual,
        RiskCategory.SelfHarm
    ],
    num_objectives=5,
)

def test_chat_target(query: str) -> str:
    return "I am a simple AI assistant that follows ethical guidelines. I'm sorry, Dave. I'm afraid I can't do that."
async def main():
    red_team_result = await red_team_agent.scan(target=test_chat_target)

asyncio.run(main())
