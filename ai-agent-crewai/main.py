from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatOpenAI

# Load dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_llm = ChatOpenAI(model_name='gpt-3.5-turbo', api_key=openai_api_key)

# Agent Organization of Events
event_research_agent = Agent(
    role='Cultural Events Researcher',
    goal='Identify the most relevant cultural and seasonal events based on user interests.',
    backstory='You are a culture and events expert with extensive knowledge of festivals, art exhibitions, and seasonal celebrations. Your mission is to discover events that offer authentic and enriching experiences.',
    llm=openai_llm
)

itinerary_planning_agent = Agent(
    role='Itinerary Planner',
    goal='Create personalized itineraries that integrate the identified events, optimizing the user experience.',
    backstory='With your logistics and travel planning skills, you can turn event research into a detailed itinerary, taking into account location, dates, and user preferences to ensure an unforgettable experience.',
    llm=openai_llm
)

# Activities for organization of events
activity_search_events = Task(
    description='''
    Identify cultural and seasonal events that match the interests and availability of the user outlined in the tags. Your final answer should be a list of recommended events, with details about each one, including dates, locations, and a brief description of the event.
    
    <event>
    - Availability: February 16-18, 2025
    - Events of interest: Classical music concert
    </event>
    ''',
    agent=event_research_agent
)

activity_planning_itinerary = Task(
    description='''
    Based on the identified events, create a detailed itinerary that optimizes the user’s trip. Include transportation recommendations, accommodations, and local tips. The final result should be a complete travel plan, with a day-by-day schedule.
    ''',
    agent=itinerary_planning_agent
)

# Creating a team with CrewAI
team = Crew(
    agent=[event_research_agent, itinerary_planning_agent],
    activities=[activity_search_events, activity_planning_itinerary],
    verbose=True
)

# Start the team job
result = team.kickoff()
