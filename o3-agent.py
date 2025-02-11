import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Initialize enhanced search tools
search_tool = SerperDevTool()
website_tool = WebsiteSearchTool()

def get_llm(use_gpt=True):
    """Get the specified language model"""
    if use_gpt:
        return ChatOpenAI(
            model_name="o3-mini-2025-01-31",
        )
    return Ollama(
        model="deepseek-r1:latest",
        base_url="http://localhost:11434",
        temperature=0.7
    )

def create_agents(use_gpt=True):
    """Create specialized research and analysis agents"""
    llm = get_llm(use_gpt)
    
    deep_researcher = Agent(
        role='Deep Research Specialist',
        goal='Conduct comprehensive internet research and data gathering',
        backstory="""Expert at conducting deep, thorough research across multiple sources. 
        Skilled at finding hard-to-locate information and connecting disparate data points. 
        Specializes in complex research tasks that would typically take hours or days.""",
        tools=[search_tool, website_tool],
        llm=llm,
        verbose=True,
        max_iter=100,          # Increased iteration limit
        allow_delegation=False, # Prevent unnecessary delegations
        max_rpm=50,            # Rate limit for API calls
        max_retry_limit=3      # Allow retries on failures
    )
    
    analyst = Agent(
        role='Research Analyst',
        goal='Analyze and synthesize complex research findings',
        backstory="""Expert analyst skilled at processing large amounts of information,
        identifying patterns, and drawing meaningful conclusions. Specializes in turning
        raw research into actionable insights.""",
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=75,
        allow_delegation=False,
        max_rpm=30,
        max_retry_limit=2
    )
    
    report_writer = Agent(
        role='Research Report Writer',
        goal='Create comprehensive, well-structured research reports',
        backstory="""Expert at transforming complex research and analysis into 
        clear, actionable reports. Skilled at maintaining detail while ensuring 
        accessibility and practical value.""",
        llm=llm,
        verbose=True,
        max_iter=50,
        allow_delegation=False,
        max_rpm=20,
        max_retry_limit=2
    )
    
    return deep_researcher, analyst, report_writer

def create_tasks(researcher, analyst, writer, research_query):
    """Create research tasks with clear objectives"""
    deep_research_task = Task(
        description=f"""Conduct focused research on: {research_query}
        
        Step-by-step approach:
        1. Initial broad search to identify key sources
        2. Deep dive into most relevant sources
        3. Extract specific details and evidence
        4. Verify key findings across sources
        5. Document sources and findings clearly
        
        Keep focused on specific, verified information.""",
        agent=researcher,
        expected_output="Detailed research findings with verified sources"
    )
    
    analysis_task = Task(
        description=f"""Analyze the research findings about {research_query}:
        
        Follow these steps:
        1. Review and categorize all findings
        2. Identify main themes and patterns
        3. Evaluate source credibility
        4. Note any inconsistencies
        5. Summarize key insights
        
        Focus on clear, actionable analysis.""",
        agent=analyst,
        context=[deep_research_task],
        expected_output="Clear analysis of findings with key insights"
    )
    
    report_task = Task(
        description=f"""Create a structured report about {research_query}:
        
        Include:
        1. Executive summary (2-3 paragraphs)
        2. Key findings (bullet points)
        3. Supporting evidence
        4. Conclusions
        5. References
        
        Keep it clear and focused.""",
        agent=writer,
        context=[deep_research_task, analysis_task],
        expected_output="Concise, well-structured report"
    )
    
    return [deep_research_task, analysis_task, report_task]

def create_crew(agents, tasks):
    """Create a crew with optimal settings"""
    return Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        max_rpm=100,  # Overall crew rate limit
        process="sequential"
    )

def main():
    print("\n🔍 Welcome to Deep Research Crew!")
    print("\nAvailable Models:")
    print("1. OpenAI o3-mini (Requires API key)")
    print("2. Local DeepSeek-r1 (Requires Ollama)")
    
    use_gpt = input("\nUse OpenAI o3-mini? (yes/no): ").lower() == 'yes'
    
    if not use_gpt:
        print("\nUsing Ollama with DeepSeek-r1")
        print("Ensure Ollama is running: ollama run deepseek-r1:latest")
    
    query = input("\nWhat would you like researched? (Be specific): ")
    
    try:
        researcher, analyst, writer = create_agents(use_gpt)
        tasks = create_tasks(researcher, analyst, writer, query)
        crew = create_crew([researcher, analyst, writer], tasks)
        
        print("\n🔍 Starting deep research process...")
        result = crew.kickoff()
        
        print("\n📊 Research Report:")
        print("==================")
        print(result)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if use_gpt:
            print("\nTip: Check your OpenAI API key")
        else:
            print("\nTip: Ensure Ollama is running with deepseek-r1:latest")
            print("Run: ollama run deepseek-r1:latest")

if __name__ == "__main__":
    main()