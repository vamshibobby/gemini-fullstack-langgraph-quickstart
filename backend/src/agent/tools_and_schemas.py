from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


class StructuredDataCheck(BaseModel):
    is_structured_query: bool = Field(
        description="Whether the user's query is asking for structured data retrieval (tables, lists, comparisons, statistics, etc.)"
    )
    reasoning: str = Field(
        description="Explanation of why this is or isn't a structured data query"
    )
    suggested_format: str = Field(
        description="If structured, suggest the best format: 'table', 'list', 'comparison', 'chart'"
    )


class TableData(BaseModel):
    headers: List[str] = Field(
        description="Column headers for the table"
    )
    rows: List[List[str]] = Field(
        description="Table rows, each row is a list of cell values"
    )
    title: str = Field(
        description="Title for the table"
    )
    caption: str = Field(
        description="Brief description of what the table shows"
    )


class VisualizationFeasibility(BaseModel):
    can_visualize: bool = Field(
        description="Whether the table data can be effectively visualized"
    )
    visualization_type: str = Field(
        description="Type of visualization: 'comparison', 'trend', 'distribution', or 'relationship'"
    )
    reasoning: str = Field(
        description="Explanation for why this visualization type was chosen"
    )
    data_characteristics: str = Field(
        description="Key characteristics of the data that support this visualization choice"
    )


class ChartTypeSelection(BaseModel):
    chart_type: str = Field(
        description="Specific chart type chosen (e.g., 'bar', 'line', 'scatter', 'histogram')"
    )
    reasoning: str = Field(
        description="Why this chart type is best for the data and visualization goal"
    )
    alternative_charts: List[str] = Field(
        description="Other chart types that could work but are less optimal"
    )


class ChartFramework(BaseModel):
    x_axis: str = Field(
        description="Column name or variable for x-axis"
    )
    y_axis: str = Field(
        description="Column name or variable for y-axis"
    )
    color_column: str = Field(
        description="Column name for color grouping (if applicable, else 'none')"
    )
    size_column: str = Field(
        description="Column name for size variation (if applicable, else 'none')"
    )
    title: str = Field(
        description="Chart title"
    )
    x_label: str = Field(
        description="X-axis label"
    )
    y_label: str = Field(
        description="Y-axis label"
    )
    chart_notes: str = Field(
        description="Additional notes or specifications for the chart"
    )


class CodeGeneration(BaseModel):
    python_code: str = Field(
        description="Complete Python code using matplotlib/seaborn to create the visualization"
    )
    required_imports: List[str] = Field(
        description="List of required import statements"
    )
    code_explanation: str = Field(
        description="Brief explanation of what the code does"
    )
    data_assumptions: str = Field(
        description="Any assumptions made about the data structure"
    )
