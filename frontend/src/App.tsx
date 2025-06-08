import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";

export default function App() {
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);

  const thread = useStream<{
    messages: Message[];
    initial_search_query_count: number;
    max_research_loops: number;
    reasoning_model: string;
  }>({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "http://localhost:8123",
    assistantId: "agent",
    messagesKey: "messages",
    onFinish: (event: any) => {
      console.log("Stream finished:", event);
    },
    onUpdateEvent: (event: any) => {
      let processedEvent: ProcessedEvent | null = null;
      if (event.structured_data_check) {
        processedEvent = {
          title: "Checking Query Type",
          data: event.structured_data_check.is_structured_query 
            ? `Structured data query detected - Format: ${event.structured_data_check.suggested_format || 'table'}`
            : "Non-structured query detected",
        };
      } else if (event.generate_query) {
        processedEvent = {
          title: "Generating Search Queries",
          data: event.generate_query.query_list ? event.generate_query.query_list.join(", ") : "Generating queries...",
        };
      } else if (event.web_research) {
        const sources = event.web_research.sources_gathered || [];
        const numSources = sources.length;
        const uniqueLabels = [
          ...new Set(sources.map((s: any) => s.label).filter(Boolean)),
        ];
        const exampleLabels = uniqueLabels.slice(0, 3).join(", ");
        processedEvent = {
          title: "Web Research",
          data: `Gathered ${numSources} sources. Related to: ${
            exampleLabels || "N/A"
          }.`,
        };
      } else if (event.reflection_structured) {
        processedEvent = {
          title: "Analyzing Structured Data",
          data: event.reflection_structured.is_sufficient
            ? "Sufficient structured data found, creating visualization."
            : `Need more structured data, searching for ${event.reflection_structured.follow_up_queries ? event.reflection_structured.follow_up_queries.join(", ") : "additional data"}`,
        };
      } else if (event.data_visualizer) {
        processedEvent = {
          title: "Preparing Data Structure",
          data: "Converting research findings into structured table format for analysis.",
        };
      } else if (event.visualization_feasibility) {
        processedEvent = {
          title: "Analyzing Visualization Options",
          data: event.visualization_feasibility.can_visualize
            ? `Can create ${event.visualization_feasibility.visualization_type} visualization`
            : "Data not suitable for visualization",
        };
      } else if (event.chart_type_selector) {
        processedEvent = {
          title: "Selecting Chart Type",
          data: `Best chart type: ${event.chart_type_selector.chart_type}`,
        };
      } else if (event.chart_framework) {
        processedEvent = {
          title: "Designing Chart Framework",
          data: `Setting up axes and variables for ${event.chart_framework.chart_spec?.title || 'chart'}`,
        };
      } else if (event.code_generator) {
        processedEvent = {
          title: "Generating Visualization Code",
          data: "Creating Python code for chart generation.",
        };
      } else if (event.chart_renderer) {
        processedEvent = {
          title: "Rendering Chart",
          data: event.chart_renderer.chart_image_url 
            ? "Chart successfully generated and saved"
            : "Executing Python code to create visualization",
        };
        hasFinalizeEventOccurredRef.current = true;
      } else if (event.non_structured_response) {
        processedEvent = {
          title: "Query Not Supported",
          data: "This query is not suitable for structured data visualization.",
        };
        hasFinalizeEventOccurredRef.current = true;
      }
      if (processedEvent) {
        setProcessedEventsTimeline((prevEvents) => [
          ...prevEvents,
          processedEvent!,
        ]);
      }
    },
  });

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [thread.messages]);

  useEffect(() => {
    if (
      hasFinalizeEventOccurredRef.current &&
      !thread.isLoading &&
      thread.messages.length > 0
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  const handleSubmit = useCallback(
    (submittedInputValue: string, effort: string, model: string) => {
      if (!submittedInputValue.trim()) return;
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;

      // convert effort to, initial_search_query_count and max_research_loops
      // low means max 1 loop and 1 query
      // medium means max 3 loops and 3 queries
      // high means max 10 loops and 5 queries
      let initial_search_query_count = 0;
      let max_research_loops = 0;
      switch (effort) {
        case "low":
          initial_search_query_count = 1;
          max_research_loops = 1;
          break;
        case "medium":
          initial_search_query_count = 3;
          max_research_loops = 3;
          break;
        case "high":
          initial_search_query_count = 5;
          max_research_loops = 10;
          break;
      }

      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];
      thread.submit({
        messages: newMessages,
        initial_search_query_count: initial_search_query_count,
        max_research_loops: max_research_loops,
        reasoning_model: model,
      });
    },
    [thread]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
    window.location.reload();
  }, [thread]);

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="flex-1 flex flex-col overflow-hidden max-w-4xl mx-auto w-full">
        <div
          className={`flex-1 overflow-y-auto ${
            thread.messages.length === 0 ? "flex" : ""
          }`}
        >
          {thread.messages.length === 0 ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={thread.isLoading}
              onCancel={handleCancel}
            />
          ) : (
            <ChatMessagesView
              messages={thread.messages}
              isLoading={thread.isLoading}
              scrollAreaRef={scrollAreaRef}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
              liveActivityEvents={processedEventsTimeline}
              historicalActivities={historicalActivities}
            />
          )}
        </div>
      </main>
    </div>
  );
}
