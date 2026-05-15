# Azure imports
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy
from pyrit.prompt_target import OpenAIChatTarget
import httpx
import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Azure AI Project Information (dict form so scan results are published to Foundry)
azure_ai_project = {
    "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", "abf1dfad-18ff-4e4e-a394-da4334202532"),
    "resource_group_name": os.getenv("AZURE_RESOURCE_GROUP", "rg-techworkshop-l300-ai-agents"),
    "project_name": os.getenv("FOUNDRY_PROJECT_NAME", "proj-kcvzmsw6lrrw4"),
}

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

# Configuration for Azure OpenAI model using managed identity
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")

gpt_endpoint = os.environ.get("gpt_endpoint").rstrip("/")

chat_target = OpenAIChatTarget(
    model_name=os.environ.get("gpt_deployment"),
    endpoint=f"{gpt_endpoint}/openai/v1/",
    api_key=token_provider,
)

async def main():
    red_team_result = await red_team_agent.scan(target=chat_target)

asyncio.run(main())
