from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import requests

load_dotenv()

# ----------------------------
# LLM
# ----------------------------
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# ----------------------------
# TOOL
# ----------------------------
@tool
def time_in_country(country: str) -> str:
    """
    Retourne l'heure actuelle d'un pays (ex: France, Japan).
    """

    if not country or not country.strip():
        return "Veuillez fournir un pays valide."

    country = country.strip()

    # ----------------------------
    # 1. Pays → timezone via REST Countries
    # ----------------------------
    try:
        res = requests.get(
            f"https://restcountries.com/v3.1/name/{country}",
            timeout=10
        )
        data = res.json()
    except Exception:
        return "Erreur API pays."

    if not isinstance(data, list) or len(data) == 0:
        return f"Pays introuvable: {country}"

    country_data = data[0]
    country_name = country_data.get("name", {}).get("common", country)

    timezones = country_data.get("timezones", [])
    if not timezones:
        return f"Aucun fuseau horaire pour {country_name}"

    tz = timezones[0]

    # ----------------------------
    # 2. TIME API (timeapi.io)
    # ----------------------------
    try:
        url = f"https://timeapi.io/api/Time/current/zone?timeZone={tz}"
        r = requests.get(url, timeout=10)
        t = r.json()
    except Exception:
        return f"Erreur API time pour {tz}"

    # ----------------------------
    # 3. RESULTAT
    # ----------------------------
    return f"Heure actuelle en {country_name} ({tz}) : {t['dateTime']}"


tools = [time_in_country]

# ----------------------------
# AGENT
# ----------------------------
agent = create_react_agent(llm, tools)

# ----------------------------
# LOOP
# ----------------------------
while True:
    question = input("\nQuestion: ")
    if not question.strip():
        continue

    result = agent.invoke({
        "messages": [("user", question)]
    })

    print("\nFinal:", result["messages"][-1].content)