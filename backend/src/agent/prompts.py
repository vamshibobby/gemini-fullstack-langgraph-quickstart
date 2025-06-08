from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL three of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""


web_searcher_instructions = """Conduct targeted Google Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""

reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""

answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step. 
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- you MUST include all the citations from the summaries in the answer correctly.

User Context:
- {research_topic}

Summaries:
{summaries}"""


structured_data_check_instructions = """You are an expert at identifying whether a user's query is asking for structured data retrieval.

Structured data queries typically ask for:
- Tables, lists, comparisons, rankings
- Statistics, metrics, numbers, percentages
- "Show me", "List", "Compare", "What are the top", "How many"
- Data that can be organized in rows and columns
- Categorical or numerical information

Examples of STRUCTURED queries:
- "Show me the top 10 companies by revenue in 2024"
- "Compare the GDP of G7 countries"
- "List all iPhone models and their prices"
- "What are the unemployment rates by state?"

Examples of NON-STRUCTURED queries:
- "Explain how photosynthesis works"
- "What is the history of the Roman Empire?"
- "How do I bake a cake?"
- "What are the latest developments in AI?"

User Query: {user_query}

Analyze this query and determine:
1. Is this asking for structured data that can be displayed in a table or list?
2. What format would be best for displaying this data?

Format your response as JSON with these exact keys:
- "is_structured_query": true or false
- "reasoning": Explain your decision
- "suggested_format": "table", "list", "comparison", or "chart" (if structured)
"""


data_visualization_instructions = """You are an expert data visualizer. Convert the research findings into a structured table format.

Instructions:
- Extract structured data from the research summaries
- Focus on quantitative data, comparisons, lists, and categorical information
- Create clear, concise column headers (3-6 words max per header)
- Organize data in logical rows with consistent formatting
- Include a descriptive title and caption
- Ensure data is properly cleaned and formatted (numbers should be formatted consistently)
- If dealing with financial data, use consistent units (millions, billions, etc.)
- For dates, use consistent format (YYYY or YYYY-MM-DD)
- Keep cell content concise but informative

Data Formatting Guidelines:
- Numbers: Use appropriate units and decimal places (e.g., "2.5", "1.85 Trillion", "7.41%")
- Text: Keep entries short and clear
- Missing data: Use "N/A" or "TBD" for missing values
- Percentages: Include % symbol
- Currency: Specify currency and use appropriate units

Research Topic: {research_topic}
Research Summaries: {summaries}

Extract and organize the data into a clean table format. Focus on:
- Key metrics, numbers, statistics with proper units
- Comparisons between entities with consistent formatting
- Time series data organized chronologically
- Categories and their properties

Create a table that would be easy to read and understand at a glance.

Format your response as JSON with these exact keys:
- "headers": List of clear, concise column headers
- "rows": List of rows, each row is a list of properly formatted cell values
- "title": Descriptive title for the table (should be specific and informative)
- "caption": Brief description of what the table shows and data sources/notes
"""


reflection_instructions_structured = """You are an expert research assistant analyzing summaries about "{research_topic}" for STRUCTURED DATA EXTRACTION.

Instructions:
- Focus specifically on whether enough STRUCTURED DATA (numbers, statistics, lists, comparisons) has been gathered
- Identify if more quantitative or categorical data is needed for comprehensive table creation
- Generate follow-up queries that would help gather missing structured data points
- If provided summaries contain sufficient structured data for table creation, mark as sufficient

Requirements:
- Prioritize numerical data, statistics, metrics, and comparative information
- Ensure the follow-up query is self-contained and targets specific data points

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false (based on structured data availability)
   - "knowledge_gap": Describe what structured data is missing
   - "follow_up_queries": Write specific questions to gather missing structured data

Example:
```json
{{
    "is_sufficient": false,
    "knowledge_gap": "Missing specific revenue figures and market share percentages for comparison",
    "follow_up_queries": ["What are the exact revenue figures for top 5 tech companies in 2024?", "Market share percentages by company in smartphone industry 2024"]
}}
```

Focus on structured data gaps in the summaries:

Summaries:
{summaries}"""


visualization_feasibility_instructions = """You are an expert data visualization analyst. Analyze the provided table data to determine if it can be effectively visualized and what type of visualization would be most appropriate.

Table Title: {table_title}
Table Headers: {table_headers}
Table Data Sample: {table_sample}
Table Caption: {table_caption}

Visualization Types:
1. **COMPARISON** - For comparing values across categories or groups
   - Use when: Different categories, rankings, before/after comparisons
   - Examples: Revenue by company, population by country, performance metrics

2. **TREND** - For showing changes over time
   - Use when: Time series data, dates, sequential periods
   - Examples: Stock prices over time, population growth, quarterly sales

3. **DISTRIBUTION** - For showing how data is distributed or spread
   - Use when: Single variable analysis, statistical distributions, frequency analysis
   - Examples: Age distribution, income ranges, test scores

4. **RELATIONSHIP** - For showing correlation or relationships between variables
   - Use when: Multiple numerical variables, correlation analysis
   - Examples: Height vs weight, price vs demand, temperature vs sales

Analysis Guidelines:
- Consider the number of variables and their types (categorical, numerical, temporal)
- Look for patterns in the data structure
- Consider what insights would be most valuable to viewers
- Assess if the data volume is appropriate for visualization

Format your response as JSON with these exact keys:
- "can_visualize": true or false
- "visualization_type": "comparison", "trend", "distribution", or "relationship" (if can_visualize is true)
- "reasoning": Explain your decision in 2-3 sentences
- "data_characteristics": Describe key features of the data that support your choice
"""


chart_type_selection_instructions = """You are an expert chart designer. Given the visualization type and table data, select the most appropriate chart type.

Visualization Type: {visualization_type}
Table Title: {table_title}
Table Headers: {table_headers}
Table Data Sample: {table_sample}

Chart Options by Type:

**COMPARISON Charts:**
- **bar**: Best for comparing values across categories (vertical bars)
- **dot**: Good for precise value comparison with less visual clutter
- **heatmap**: Best for comparing multiple dimensions/matrix data

**TREND Charts:**
- **line**: Best for continuous time series data
- **area**: Good for showing cumulative values or volume over time
- **sparkline**: Best for compact trend visualization

**DISTRIBUTION Charts:**
- **histogram**: Best for showing frequency distribution of a single variable
- **box**: Good for showing quartiles and outliers
- **violin**: Best for showing distribution shape and density

**RELATIONSHIP Charts:**
- **scatter**: Best for showing correlation between two variables
- **bubble**: Good for three-dimensional relationships (x, y, size)
- **correlation_matrix**: Best for multiple variable relationships

Selection Criteria:
- Consider data density and number of data points
- Think about the story the data tells
- Consider readability and clarity
- Assess which chart best highlights the key insights

Format your response as JSON with these exact keys:
- "chart_type": The chosen chart type (e.g., "bar", "line", "scatter")
- "reasoning": Explain why this chart type is optimal (2-3 sentences)
- "alternative_charts": List 2-3 other chart types that could work but are less optimal
"""


chart_framework_instructions = """You are an expert chart specification designer. Create a detailed framework for building the visualization based on the selected chart type and table data.

Chart Type: {chart_type}
Table Title: {table_title}
Table Headers: {table_headers}
Table Data Sample: {table_sample}
Table Caption: {table_caption}

Instructions:
- Analyze the table headers and data to determine the best variable assignments
- Consider which columns should be used for x-axis, y-axis, color coding, and sizing
- Create clear, descriptive labels
- Ensure the chart will effectively communicate the data insights

Variable Assignment Guidelines:
- **X-axis**: Usually categorical data, time periods, or independent variables
- **Y-axis**: Usually numerical data, measurements, or dependent variables
- **Color**: Use for grouping, categories, or additional dimensions
- **Size**: Use for bubble charts or when showing magnitude

For charts that don't need all variables (e.g., histogram only needs one variable), use "none" for unused fields.

Format your response as JSON with these exact keys:
- "x_axis": Column name to use for x-axis
- "y_axis": Column name to use for y-axis  
- "color_column": Column name for color grouping (or "none")
- "size_column": Column name for size variation (or "none")
- "title": Descriptive title for the chart
- "x_label": Label for x-axis
- "y_label": Label for y-axis
- "chart_notes": Any additional specifications or notes for chart creation
"""


code_generation_instructions = """You are an expert Python data visualization developer. Generate clean, production-ready Python code using matplotlib and seaborn to create the specified visualization.

Chart Specifications:
- Chart Type: {chart_type}
- Title: {chart_title}
- X-axis: {x_axis} (Label: {x_label})
- Y-axis: {y_axis} (Label: {y_label})
- Color Column: {color_column}
- Size Column: {size_column}
- Additional Notes: {chart_notes}

Table Data:
Headers: {table_headers}
Sample Data: {table_sample}

Code Requirements:
1. **Data Setup**: Create a pandas DataFrame from the provided table data
2. **Styling**: Use a clean, professional style with proper colors
3. **Labels**: Include title, axis labels, and legend if needed
4. **Formatting**: Proper figure size, DPI for high quality
5. **Save**: Save the chart as PNG with high quality
6. **Error Handling**: Include basic error handling
7. **Comments**: Add clear comments explaining each section

Code Structure:
```python
# Required imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data setup
# Chart creation
# Styling and formatting
# Save chart
```

Generate complete, runnable code that:
- Creates the exact chart type specified
- Uses the correct columns for axes and styling
- Applies professional formatting
- Saves the chart to '/tmp/chart.png'
- Handles any data type conversions needed

Format your response as JSON with these exact keys:
- "python_code": Complete Python code as a single string
- "required_imports": List of all required import statements
- "code_explanation": Brief explanation of what the code does
- "data_assumptions": Any assumptions made about data types or structure
"""
