import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, WebsiteSearchTool
from langchain_openai import ChatOpenAI
import requests

# Load environment variables
load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Initialize enhanced search tools
search_tool = SerperDevTool()
website_tool = WebsiteSearchTool()

def check_ollama_availability():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/version")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def get_llm(use_gpt=True):
    """Initialize the specified language model"""   
    if use_gpt:
        return ChatOpenAI(model_name="o1-mini")
    
    # Use CrewAI's LLM class with correct provider format
    return LLM(
        model="ollama/deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        temperature=0.7
    )

def create_agents(use_gpt=True):
    """Create specialized research and analysis agents"""
    try:
        llm = get_llm(use_gpt)
        
        researcher = Agent(
            role='Deep Research Specialist',
            goal='Conduct comprehensive research and gather detailed information',
            backstory="""Expert researcher skilled at discovering hard-to-find information 
            and connecting complex data points. Specializes in thorough, detailed research.""",
            tools=[search_tool, website_tool],
            llm=llm,
            verbose=True,
            max_iter=15,
            allow_delegation=False
        )
        
        analyst = Agent(
            role='Research Analyst',
            goal='Analyze and synthesize research findings',
            backstory="""Expert analyst skilled at processing complex information and 
            identifying key patterns and insights. Specializes in clear, actionable analysis.""",
            tools=[search_tool],
            llm=llm,
            verbose=True,
            max_iter=10,
            allow_delegation=False
        )
        
        writer = Agent(
            role='Content Synthesizer',
            goal='Create clear, structured reports from analysis',
            backstory="""Expert writer skilled at transforming complex analysis into 
            clear, engaging content while maintaining technical accuracy.""",
            llm=llm,
            verbose=True,
            max_iter=8,
            allow_delegation=False
        )
        
        return researcher, analyst, writer
    except Exception as e:
        st.error(f"Error creating agents: {str(e)}")
        return None, None, None

def create_tasks(researcher, analyst, writer, topic):
    """Create research tasks with clear objectives"""
    research_task = Task(
        description=f"""Research this topic thoroughly: {topic}
        
        Follow these steps:
        1. Find reliable sources and latest information
        2. Extract key details and evidence
        3. Verify information across sources
        4. Document findings with references""",
        agent=researcher,
        expected_output="Detailed research findings with sources"
    )
    
    analysis_task = Task(
        description=f"""Analyze the research findings about {topic}:
        
        Steps:
        1. Review and categorize findings
        2. Identify patterns and trends
        3. Evaluate source credibility
        4. Note key insights""",
        agent=analyst,
        context=[research_task],
        expected_output="Analysis of findings and insights"
    )
    
    synthesis_task = Task(
        description=f"""Create a clear report on {topic}:
        
        Include:
        - Executive Summary
        - Key Findings
        - Evidence
        - Conclusions
        - Specific questions asked by the user
        - search volume, demand, search converstion
        - Top keywords
        - References""",
        agent=writer,
        context=[research_task, analysis_task],
        expected_output="Structured report with insights"
    )
    
    return [research_task, analysis_task, synthesis_task]

def run_research(topic, use_gpt):
    """Execute the research process"""
    try:
        if use_gpt and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        
        if not use_gpt and not check_ollama_availability():
            raise ConnectionError("Ollama server not running. Start with: ollama run deepseek-r1")
        
        researcher, analyst, writer = create_agents(use_gpt)
        if not all([researcher, analyst, writer]):
            raise Exception("Failed to create agents")
        
        tasks = create_tasks(researcher, analyst, writer, topic)
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=tasks,
            verbose=True
        )
        
        result = crew.kickoff()
        # Convert CrewOutput to string for consistency
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    st.set_page_config(
        page_title="Deep Research Assistant",
        page_icon="🔍",
        layout="wide"
    )

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    model_choice = st.sidebar.radio(
        "Choose Model",
        ["OpenAI o1-mini", "Local DeepSeek-r1"]
    )
    use_gpt = model_choice == "OpenAI o1-mini"

    if use_gpt and not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning("⚠️ OpenAI API key not found")
    
    if not use_gpt:
        if not check_ollama_availability():
            st.sidebar.warning("⚠️ Ollama not running. Run: `ollama pull deepseek-r1 && ollama run deepseek-r1`")
        else:
            st.sidebar.success("✅ Ollama running")

    # Main content
    st.title("🔍 Deep Research Assistant")
    st.markdown("""
    This AI-powered research assistant conducts comprehensive research on any topic.
    It uses specialized agents to research, analyze, and synthesize information.
    """)

    # Input
    query = st.text_area(
        "Research Topic",
        placeholder="Enter your research topic (be specific)...",
        help="More specific queries yield better results"
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        start_research = st.button("🚀 Start Research", type="primary")

    # Execute research
    if start_research and query:
        with st.spinner("🔍 Conducting research..."):
            result = run_research(query, use_gpt)

        if isinstance(result, str) and result.startswith("Error:"):
            st.error(result)
        else:
            st.success("✅ Research Complete!")
            
            tab1, tab2 = st.tabs(["📊 Report", "ℹ️ About"])
            
            with tab1:
                st.markdown("### Research Report")
                st.markdown("---")
                st.markdown(str(result))  # Ensure result is converted to string
                
            with tab2:
                st.markdown(f"""
                ### Process:
                1. **Research**: Comprehensive source search
                2. **Analysis**: Pattern identification
                3. **Synthesis**: Report creation
                
                **Details:**
                - Model: {model_choice}
                - Tools: Web search, content analysis
                - Method: Multi-agent collaboration
                """)
    
    st.divider()
    st.markdown("*Built with CrewAI and Streamlit*")

if __name__ == "__main__":
    main()