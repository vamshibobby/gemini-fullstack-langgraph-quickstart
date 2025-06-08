import os
import subprocess
import sys
import uuid
import tempfile
import shutil

from agent.tools_and_schemas import SearchQueryList, Reflection, StructuredDataCheck, TableData, VisualizationFeasibility, ChartTypeSelection, ChartFramework, CodeGeneration
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    reflection_instructions_structured,
    answer_instructions,
    structured_data_check_instructions,
    data_visualization_instructions,
    visualization_feasibility_instructions,
    chart_type_selection_instructions,
    chart_framework_instructions,
    code_generation_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search query for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reflection_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to Gemini 1.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Add the final answer to the messages
    return {"messages": [AIMessage(content=result.content)]}


def structured_data_check(state: OverallState, config: RunnableConfig):
    """LangGraph node that checks if the user's query is asking for structured data.
    
    Args:
        state: Current graph state containing the user's query
        config: Configuration for the runnable
    
    Returns:
        Dictionary with state update indicating if this is a structured data query
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Get the user's query from the messages
    user_query = get_research_topic(state["messages"])
    
    # Format the prompt
    formatted_prompt = structured_data_check_instructions.format(
        user_query=user_query
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=configurable.reflection_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    structured_llm = llm.with_structured_output(StructuredDataCheck)
    result = structured_llm.invoke(formatted_prompt)
    
    return {
        "is_structured_query": result.is_structured_query,
        "visualization_format": result.suggested_format,
    }


def route_based_on_query_type(state: OverallState):
    """Routes to either continue with research or return error for non-structured queries."""
    if state.get("is_structured_query", False):
        return "generate_query"
    else:
        return "non_structured_response"


def non_structured_response(state: OverallState, config: RunnableConfig):
    """Returns an error message for non-structured queries."""
    error_message = "I can only answer structured data retrieval queries. Please ask for tables, lists, comparisons, statistics, or other structured data that can be visualized."
    return {"messages": [AIMessage(content=error_message)]}


def reflection_structured(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps specifically for structured data extraction.
    
    Modified version of reflection that focuses on structured data requirements.
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reflection_model

    # Format the prompt for structured data reflection
    current_date = get_current_date()
    formatted_prompt = reflection_instructions_structured.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    
    # init Reasoning Model
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research_structured(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function for structured data research flow."""
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "data_visualizer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def data_visualizer(state: OverallState, config: RunnableConfig):
    """LangGraph node that converts research findings into structured table format.
    
    Args:
        state: Current graph state containing research summaries
        config: Configuration for the runnable
    
    Returns:
        Dictionary with state update containing the structured table data
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = data_visualization_instructions.format(
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    structured_llm = llm.with_structured_output(TableData)
    result = structured_llm.invoke(formatted_prompt)
    
    # Create a well-formatted HTML table
    html_table = f"""
<div style="margin: 20px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <h2 style="color: #f1f5f9; margin-bottom: 16px; font-size: 24px; font-weight: 600; text-align: center;">{result.title}</h2>
    
    <div style="overflow-x: auto; border-radius: 12px; border: 1px solid #4a5568; background: #2d3748; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <thead>
                <tr style="background: linear-gradient(90deg, #4a5568 0%, #2d3748 100%);">
                    {' '.join(f'<th style="padding: 16px 20px; text-align: left; font-weight: 600; color: #e2e8f0; border-bottom: 2px solid #4a5568; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">{header}</th>' for header in result.headers)}
                </tr>
            </thead>
            <tbody>
                {' '.join(f'''<tr style="border-bottom: 1px solid #4a5568; transition: background-color 0.2s ease;" onmouseover="this.style.backgroundColor='#374151'" onmouseout="this.style.backgroundColor='transparent'">
                    {' '.join(f'<td style="padding: 14px 20px; color: #f7fafc; font-size: 14px; line-height: 1.5;">{str(cell)}</td>' for cell in row)}
                </tr>''' for row in result.rows)}
            </tbody>
        </table>
    </div>
    
    <p style="margin-top: 16px; font-style: italic; color: #a0aec0; font-size: 13px; text-align: center; line-height: 1.4;">{result.caption}</p>
</div>
"""
    
    # Also create a clean markdown version as fallback
    markdown_table = f"## {result.title}\n\n"
    
    # Create properly formatted markdown table with better spacing
    if result.headers and result.rows:
        # Calculate column widths for better alignment
        col_widths = []
        for i, header in enumerate(result.headers):
            max_width = len(str(header))
            for row in result.rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max(max_width, 8))  # Minimum width of 8
        
        # Create header row
        header_row = "| " + " | ".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(result.headers)) + " |"
        separator_row = "| " + " | ".join("-" * width for width in col_widths) + " |"
        
        markdown_table += header_row + "\n" + separator_row + "\n"
        
        # Add data rows
        for row in result.rows:
            row_cells = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    row_cells.append(f"{str(cell):<{col_widths[i]}}")
                else:
                    row_cells.append(str(cell))
            row_str = "| " + " | ".join(row_cells) + " |"
            markdown_table += row_str + "\n"
    
    markdown_table += f"\n*{result.caption}*"
    
    # Store both the structured data and the HTML table for later use
    return {
        "structured_data": {
            "headers": result.headers,
            "rows": result.rows,
            "title": result.title,
            "caption": result.caption
        },
        "table_html": html_table  # Store the HTML table for final display
    }


def visualization_feasibility(state: OverallState, config: RunnableConfig):
    """LangGraph node that determines if the table data can be visualized and what type.
    
    Args:
        state: Current graph state containing the structured table data
        config: Configuration for the runnable
    
    Returns:
        Dictionary with state update containing visualization feasibility and type
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Get table data from state
    table_data = state.get("structured_data", {})
    if not table_data:
        return {
            "can_visualize": False,
            "visualization_type": "none"
        }
    
    # Prepare data for analysis (first 3 rows as sample)
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    title = table_data.get("title", "")
    caption = table_data.get("caption", "")
    
    # Take first 3 rows as sample for analysis
    sample_rows = rows[:3] if len(rows) > 3 else rows
    
    # Format the prompt
    formatted_prompt = visualization_feasibility_instructions.format(
        table_title=title,
        table_headers=", ".join(headers),
        table_sample=str(sample_rows),
        table_caption=caption
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=configurable.reflection_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    structured_llm = llm.with_structured_output(VisualizationFeasibility)
    result = structured_llm.invoke(formatted_prompt)
    
    return {
        "can_visualize": result.can_visualize,
        "visualization_type": result.visualization_type if result.can_visualize else "none"
    }


def chart_type_selector(state: OverallState, config: RunnableConfig):
    """LangGraph node that selects the best chart type for the visualization.
    
    Args:
        state: Current graph state containing visualization type and table data
        config: Configuration for the runnable
    
    Returns:
        Dictionary with state update containing the selected chart type
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Get required data from state
    visualization_type = state.get("visualization_type", "")
    table_data = state.get("structured_data", {})
    
    if not table_data or visualization_type == "none":
        return {"chart_type": "none"}
    
    # Prepare data for analysis
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    title = table_data.get("title", "")
    sample_rows = rows[:3] if len(rows) > 3 else rows
    
    # Format the prompt
    formatted_prompt = chart_type_selection_instructions.format(
        visualization_type=visualization_type,
        table_title=title,
        table_headers=", ".join(headers),
        table_sample=str(sample_rows)
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=configurable.reflection_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    structured_llm = llm.with_structured_output(ChartTypeSelection)
    result = structured_llm.invoke(formatted_prompt)
    
    return {"chart_type": result.chart_type}


def chart_framework(state: OverallState, config: RunnableConfig):
    """LangGraph node that creates a framework for building the visualization.
    
    Args:
        state: Current graph state containing chart type and table data
        config: Configuration for the runnable
    
    Returns:
        Dictionary with state update containing the chart specification
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Get required data from state
    chart_type = state.get("chart_type", "")
    table_data = state.get("structured_data", {})
    
    if not table_data or chart_type == "none":
        return {"chart_spec": {}}
    
    # Prepare data for analysis
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    title = table_data.get("title", "")
    caption = table_data.get("caption", "")
    sample_rows = rows[:3] if len(rows) > 3 else rows
    
    # Format the prompt
    formatted_prompt = chart_framework_instructions.format(
        chart_type=chart_type,
        table_title=title,
        table_headers=", ".join(headers),
        table_sample=str(sample_rows),
        table_caption=caption
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=configurable.reflection_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    structured_llm = llm.with_structured_output(ChartFramework)
    result = structured_llm.invoke(formatted_prompt)
    
    return {
        "chart_spec": {
            "x_axis": result.x_axis,
            "y_axis": result.y_axis,
            "color_column": result.color_column,
            "size_column": result.size_column,
            "title": result.title,
            "x_label": result.x_label,
            "y_label": result.y_label,
            "chart_notes": result.chart_notes
        }
    }


def code_generator(state: OverallState, config: RunnableConfig):
    """LangGraph node that generates Python code for creating the visualization.
    
    Args:
        state: Current graph state containing chart specifications and table data
        config: Configuration for the runnable
    
    Returns:
        Dictionary with state update containing the generated Python code
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Get required data from state
    chart_type = state.get("chart_type", "")
    chart_spec = state.get("chart_spec", {})
    table_data = state.get("structured_data", {})
    
    if not table_data or chart_type == "none" or not chart_spec:
        return {"chart_code": ""}
    
    # Prepare data for code generation
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    sample_rows = rows[:3] if len(rows) > 3 else rows
    
    # Format the prompt
    formatted_prompt = code_generation_instructions.format(
        chart_type=chart_type,
        chart_title=chart_spec.get("title", ""),
        x_axis=chart_spec.get("x_axis", ""),
        x_label=chart_spec.get("x_label", ""),
        y_axis=chart_spec.get("y_axis", ""),
        y_label=chart_spec.get("y_label", ""),
        color_column=chart_spec.get("color_column", "none"),
        size_column=chart_spec.get("size_column", "none"),
        chart_notes=chart_spec.get("chart_notes", ""),
        table_headers=", ".join(headers),
        table_sample=str(sample_rows)
    )
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=configurable.answer_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    structured_llm = llm.with_structured_output(CodeGeneration)
    result = structured_llm.invoke(formatted_prompt)
    
    return {"chart_code": result.python_code}


def chart_renderer(state: OverallState, config: RunnableConfig):
    """LangGraph node that executes the Python code to generate chart images.
    
    Args:
        state: Current graph state containing the Python code and table data
        config: Configuration for the runnable
    
    Returns:
        Dictionary with state update containing the chart image URL
    """
    chart_code = state.get("chart_code", "")
    table_data = state.get("structured_data", {})
    
    if not chart_code or not table_data:
        return {"chart_image_url": ""}
    
    # Create a unique filename for the chart
    chart_id = str(uuid.uuid4())
    chart_filename = f"chart_{chart_id}.png"
    chart_path = os.path.join("static", "charts", chart_filename)
    
    # Prepare the table data as CSV-like structure for the Python code
    headers = table_data.get("headers", [])
    rows = table_data.get("rows", [])
    
    # Create a temporary Python file with the complete code
    temp_dir = tempfile.mkdtemp()
    try:
        temp_python_file = os.path.join(temp_dir, "chart_generator.py")
        
        # Prepare the complete Python code with data injection
        complete_code = f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set matplotlib backend for headless execution
import matplotlib
matplotlib.use('Agg')

# Prepare the data
headers = {headers}
rows = {rows}

# Create DataFrame
df = pd.DataFrame(rows, columns=headers)

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname('{chart_path}'), exist_ok=True)

# Configure matplotlib for better appearance
plt.style.use('default')
sns.set_palette("husl")

# User-generated code starts here
{chart_code}

# Save the chart
plt.tight_layout()
plt.savefig('{chart_path}', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Chart saved successfully")
"""
        
        # Write the code to the temporary file
        with open(temp_python_file, 'w') as f:
            f.write(complete_code)
        
        # Execute the Python code in a subprocess
        try:
            result = subprocess.run(
                [sys.executable, temp_python_file],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=os.getcwd()  # Run from the backend directory
            )
            
            if result.returncode == 0:
                # Chart was created successfully
                chart_url = f"/static/charts/{chart_filename}"
                
                # Get the table HTML from state (stored by data_visualizer)
                table_html = state.get("table_html", "")
                
                # Create final content with table and chart URL
                final_content = table_html
                if chart_url:
                    final_content += f"\n\nCHART_IMAGE_URL: {chart_url}"
                
                return {
                    "chart_image_url": chart_url,
                    "messages": [AIMessage(content=final_content)]
                }
            else:
                print(f"Chart generation failed: {result.stderr}")
                return {"chart_image_url": ""}
                
        except subprocess.TimeoutExpired:
            print("Chart generation timed out")
            return {"chart_image_url": ""}
        except Exception as e:
            print(f"Error executing chart code: {str(e)}")
            return {"chart_image_url": ""}
            
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def route_visualization_path(state: OverallState):
    """Routes based on whether visualization is possible."""
    if state.get("can_visualize", False):
        return "chart_type_selector"
    else:
        return "final_summary"


def final_summary(state: OverallState, config: RunnableConfig):
    """Creates the final message that includes table and optionally chart image URL."""
    
    # Get the table HTML from state (stored by data_visualizer)
    table_html = state.get("table_html", "")
    chart_image_url = state.get("chart_image_url", "")
    
    if not table_html:
        return {"messages": [AIMessage(content="No structured data was generated.")]}
    
    # Create the final message content
    final_content = table_html
    
    # Add chart URL if available
    if chart_image_url:
        final_content += f"\n\nCHART_IMAGE_URL: {chart_image_url}"
    
    return {"messages": [AIMessage(content=final_content)]}


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define all nodes
builder.add_node("structured_data_check", structured_data_check)
builder.add_node("non_structured_response", non_structured_response)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection_structured", reflection_structured)
builder.add_node("data_visualizer", data_visualizer)
builder.add_node("visualization_feasibility", visualization_feasibility)
builder.add_node("chart_type_selector", chart_type_selector)
builder.add_node("chart_framework", chart_framework)
builder.add_node("code_generator", code_generator)
builder.add_node("chart_renderer", chart_renderer)
builder.add_node("final_summary", final_summary)

# Set the entrypoint as structured_data_check
builder.add_edge(START, "structured_data_check")

# Route based on query type
builder.add_conditional_edges(
    "structured_data_check", 
    route_based_on_query_type, 
    ["generate_query", "non_structured_response"]
)

# Non-structured queries end here
builder.add_edge("non_structured_response", END)

# Structured data flow continues
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)

# Web research goes to structured reflection
builder.add_edge("web_research", "reflection_structured")

# Evaluate structured research
builder.add_conditional_edges(
    "reflection_structured", evaluate_research_structured, ["web_research", "data_visualizer"]
)

# After data visualization, check visualization feasibility
builder.add_edge("data_visualizer", "visualization_feasibility")

# Route based on visualization feasibility
builder.add_conditional_edges(
    "visualization_feasibility", 
    route_visualization_path, 
    ["chart_type_selector", "final_summary"]
)

# Visualization pipeline
builder.add_edge("chart_type_selector", "chart_framework")
builder.add_edge("chart_framework", "code_generator")
builder.add_edge("code_generator", "chart_renderer")
builder.add_edge("chart_renderer", END)

# Final summary for non-chart cases
builder.add_edge("final_summary", END)

graph = builder.compile(name="pro-search-agent")
